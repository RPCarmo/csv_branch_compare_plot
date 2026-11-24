#!/usr/bin/env python3
"""
csv_branch_compare.py - compare, as plots, the CSV files changed in a git branch.

Run it from inside your local clone of the repository, with an environment that has
pandas and matplotlib (e.g. the repository's pixi environment):
    pixi run python csv_branch_compare.py

In the window, type a branch name and press Load:
    my-branch          -> local branch
    origin/my-branch   -> branch on the remote
Either way, all branches of the remote are fetched first, using your normal git
credentials (SSH keys, credential manager, etc.).
Then pick a changed CSV file and the X and Y columns. The "before" and "after"
curves are plotted together; nothing is saved to disk.

By default, "before" is the file where the branch split off from the branch it was
created from (shown in the window), so the plot shows only the changes introduced by
the branch. Git doesn't record that branch, so it is inferred from the history: see
`find_base`. Fill in the "Base" field to compare against any other branch, tag or
commit instead (e.g. my-branch~1 to see only the last commit). Only committed changes
are compared.

Requires: git on PATH, Python 3.9+, pandas, matplotlib (Tkinter ships with Python)
"""
import io
import os
import subprocess
import sys
import textwrap
import threading
import warnings
from dataclasses import dataclass

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter import font as tkfont

    import pandas as pd
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError as import_error:
    sys.exit(
        f"Missing dependency: '{import_error.name}'.\n"
        "This script needs pandas, matplotlib and Tkinter. Run it with an environment "
        "that has them, e.g. from the repository folder:\n"
        f"    pixi run python {sys.argv[0]}"
    )

DEFAULT_REMOTE = "origin"  # used to find the default branch name for local branches


# --------------------------------------------------------------------------- git
class GitError(RuntimeError):
    pass


def run_git(*args, capture=True, input=None):
    """Run a git command in the current directory. With capture=False, output
    and any credential prompts go straight to the terminal."""
    try:
        proc = subprocess.run(
            ["git", *args],
            input=input,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError:
        raise GitError("'git' was not found on your PATH.")
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip() if capture else ""
        raise GitError(f"'git {' '.join(args)}' failed" + (f":\n{err}" if err else "."))
    return proc.stdout if capture else b""


def git_text(*args, input=None):
    return run_git(*args, input=input).decode("utf-8", "replace").strip()


def ref_exists(ref):
    try:
        run_git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        return True
    except GitError:
        return False


def default_branch(remote):
    """Name of the repo's default branch (e.g. main/master)."""
    if remote:
        try:
            return git_text("symbolic-ref", "--short", f"refs/remotes/{remote}/HEAD").split("/", 1)[1]
        except GitError:
            pass
        try:
            for line in git_text("ls-remote", "--symref", remote, "HEAD").splitlines():
                if line.startswith("ref:"):
                    return line.split()[1].removeprefix("refs/heads/")
        except GitError:
            pass
    for name in ("main", "master"):
        if ref_exists(f"refs/heads/{name}") or (remote and ref_exists(f"refs/remotes/{remote}/{name}")):
            return name
    raise GitError("Could not determine the default branch (main/master).")


def fetch_all(remote):
    """Fetch every branch of the remote, so the candidate base branches are current."""
    print(f"Fetching all branches from '{remote}'...")
    run_git("fetch", remote, f"+refs/heads/*:refs/remotes/{remote}/*", capture=False)


def resolve_branch(name):
    """Return (head_ref, branch_name, remote) for the branch the user typed.
    'origin/x' -> branch x on origin; plain 'x' -> local branch x."""
    remotes = git_text("remote").split()
    remote, _, branch = name.partition("/")

    if branch and remote in remotes:  # upstream branch
        fetch_all(remote)
        head_ref = f"refs/remotes/{remote}/{branch}"
        if not ref_exists(head_ref):
            raise GitError(f"Branch '{branch}' doesn't exist on '{remote}'.")
        return head_ref, branch, remote

    # local branch
    if not ref_exists(f"refs/heads/{name}"):
        raise GitError(f"Branch '{name}' doesn't exist locally.\n"
                       f"To search it on the remote, write: origin/{name}")
    remote = DEFAULT_REMOTE if DEFAULT_REMOTE in remotes else (remotes[0] if remotes else None)
    if remote:
        fetch_all(remote)
    return f"refs/heads/{name}", name, remote


def short_ref(ref):
    return ref.removeprefix("refs/heads/").removeprefix("refs/remotes/")


def find_base(head_ref, branch, remote):
    """Return (base_ref, base_label): the branch `head_ref` was most likely created from.

    The fork point is the first commit in the branch's first-parent history that
    some other branch also contains. Every branch containing it yields the same
    merge-base, so the choice among them only affects the label: the default
    branch wins, then the one whose tip is closest to the fork point. Branches that
    already contain the whole branch (e.g. it was merged into them) and the
    branch's own local/remote copies are not candidates.
    """
    head_sha = git_text("rev-parse", head_ref)
    remote_prefix = f"refs/remotes/{remote}/" if remote else None

    def is_own_copy(ref):
        own = (f"refs/heads/{branch}", f"{remote_prefix}{branch}")
        return ref in own or ref.endswith("/HEAD")

    def branch_refs(*extra):
        refs = git_text("for-each-ref", "--format=%(refname)", *extra, "refs/heads", "refs/remotes")
        return [r for r in refs.splitlines() if not is_own_copy(r)]

    descendants = set(branch_refs("--contains", head_sha))
    others = [r for r in branch_refs() if r not in descendants]
    stdin = "".join(f"^{r}\n" for r in others).encode()
    unique = git_text("rev-list", "--first-parent", "--stdin", head_sha, input=stdin).splitlines()
    try:
        fork = git_text("rev-parse", f"{unique[-1]}^" if unique else head_sha)
        candidates = [r for r in branch_refs("--contains", fork) if r not in descendants]
    except GitError:  # The branch's history doesn't meet any other branch.
        candidates = []
    if not candidates:
        raise GitError(f"Could not find the branch '{branch}' was created from.")

    default = default_branch(remote)
    preferred = [r for r in (f"{remote_prefix}{default}", f"refs/heads/{default}") if r in candidates]
    if preferred:
        base_ref = preferred[0]
    else:
        def distance(ref):
            return int(git_text("rev-list", "--count", f"{fork}..{ref}"))
        # On ties, prefer remote branches: local copies are often stale.
        base_ref = min(candidates, key=lambda r: (distance(r), r.startswith("refs/heads/"), r))
    return base_ref, short_ref(base_ref)


@dataclass(frozen=True)
class Comparison:
    """What is being compared: the 'before' commit and the 'after' side."""

    base_rev: str
    base_label: str
    """The branch, tag or commit 'before' comes from, as shown to the user."""
    base_is_split_point: bool
    """Whether `base_rev` was inferred as the point where the branch split off."""
    head_rev: str
    head_label: str


def resolve_comparison(branch: str, base: str) -> Comparison:
    """
    Work out 'before' and 'after' from what the user typed in the window.

    An empty `base` means the point where the branch split off the branch it was created
    from. The branch is resolved (and fetched) first, so a typed remote base is current.
    """
    if not branch:
        raise GitError("Type a branch name.")
    head_ref, branch_name, remote = resolve_branch(branch)
    if base:
        if not ref_exists(base):
            raise GitError(f"Base '{base}' is not a known branch, tag or commit.")
        base_rev = git_text("rev-parse", "--verify", f"{base}^{{commit}}")
        return Comparison(base_rev, base, False, head_ref, branch)
    base_ref, base_label = find_base(head_ref, branch_name, remote)
    base_rev = git_text("merge-base", base_ref, head_ref)
    return Comparison(base_rev, base_label, True, head_ref, branch)


def changed_csvs(base_rev, head_rev):
    """Return (status, old_path, new_path) for every CSV changed between two revisions."""
    raw = run_git("diff", "--name-status", "-M", "-z", base_rev, head_rev)
    parts = [p for p in raw.decode("utf-8", "replace").split("\0") if p]
    files, i = [], 0
    while i < len(parts):
        status = parts[i][0]
        if status in "RC":  # renamed / copied: two paths
            old, new = parts[i + 1], parts[i + 2]
            i += 3
        else:
            old = new = parts[i + 1]
            i += 2
        if old.lower().endswith(".csv") or new.lower().endswith(".csv"):
            files.append((status, old, new))
    return files


def file_label(status, old, new):
    names = {"A": "added", "M": "modified", "D": "deleted",
             "R": "renamed", "C": "copied", "T": "type changed"}
    path = f"{old} -> {new}" if status in "RC" else new
    return f"{path}  ({names.get(status, status)})"


# --------------------------------------------------------------------------- data
def read_csv(data):
    if data is None:
        return None
    try:  # let pandas detect the delimiter (, ; tab ...)
        return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")


def as_plottable(series):
    """Numeric if possible, then datetime, otherwise plain text (categorical)."""
    valid = series.notna().sum()
    num = pd.to_numeric(series, errors="coerce")
    if valid and num.notna().sum() >= 0.9 * valid:
        return num
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt = pd.to_datetime(series, errors="coerce")
    if valid and dt.notna().sum() >= 0.9 * valid:
        return dt
    return series.astype(str)


# --------------------------------------------------------------------------- GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV branch compare")
        self.geometry("1150x680")
        self.minsize(800, 500)

        self.comparison = None
        self.files = []
        self.df_before = self.df_after = None
        self.file_name = ""
        self.columns = []

        self._build_controls()
        self._build_plot()
        self._show_placeholder("Enter a branch name and press Load.")

    # ------------------------------------------------------------------ layout
    def _build_controls(self):
        # The divider between the controls and the plot can be dragged; it also
        # widens automatically to fit long file and column names.
        self.panes = tk.PanedWindow(self, orient="horizontal", sashwidth=6, sashrelief="raised")
        self.panes.pack(fill="both", expand=True)
        self.side = side = ttk.Frame(self.panes, padding=12)
        # minsize keeps the panel from ever collapsing; width is its starting size
        self.panes.add(side, minsize=300, width=380, stretch="never")
        side.columnconfigure(0, weight=1)

        ttk.Label(side, text="Branch (after)").grid(row=0, column=0, sticky="w")
        row = ttk.Frame(side)
        row.grid(row=1, column=0, sticky="ew", pady=(2, 2))
        self.branch_var = tk.StringVar()
        entry = ttk.Entry(row, textvariable=self.branch_var)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda e: self.load_branch())
        entry.focus_set()
        self.load_btn = ttk.Button(row, text="Load", command=self.load_branch)
        self.load_btn.pack(side="left", padx=(6, 0))
        ttk.Label(side, text="my-branch = local, origin/my-branch = remote",
                  foreground="gray").grid(row=2, column=0, sticky="w")

        ttk.Label(side, text="Base (before) - optional").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.base_var = tk.StringVar()
        base_entry = ttk.Entry(side, textvariable=self.base_var)
        base_entry.grid(row=4, column=0, sticky="ew", pady=(2, 2))
        base_entry.bind("<Return>", lambda e: self.load_branch())
        ttk.Label(side, text="empty = the branch it was created from",
                  foreground="gray").grid(row=5, column=0, sticky="w")

        self.base_label = ttk.Label(side, text="Before (split from): -")
        self.base_label.grid(row=6, column=0, sticky="w", pady=(8, 0))

        self.status = ttk.Label(side, text="", justify="left")
        self.status.grid(row=7, column=0, sticky="ew", pady=(10, 10))
        side.bind("<Configure>", lambda e: self.status.configure(wraplength=max(e.width - 30, 100)))

        self.file_list = self._make_list(side, "Changed CSV files", first_row=8, weight=2)
        self.x_list = self._make_list(side, "X axis", first_row=10, weight=1)
        self.y_list = self._make_list(side, "Y axis", first_row=12, weight=1)
        self.file_list.bind("<<ListboxSelect>>", self.on_file_selected)
        for lb in (self.x_list, self.y_list):
            lb.bind("<<ListboxSelect>>", lambda e: self.update_plot())

        self.list_font = tkfont.Font(font=self.file_list.cget("font"))

    def _make_list(self, parent, title, first_row, weight):
        """A labelled listbox with vertical and horizontal scrollbars."""
        ttk.Label(parent, text=title).grid(row=first_row, column=0, sticky="w", pady=(8, 2))
        frame = ttk.Frame(parent)
        frame.grid(row=first_row + 1, column=0, sticky="nsew")
        parent.rowconfigure(first_row + 1, weight=weight)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        lb = tk.Listbox(frame, exportselection=False, height=5)
        ys = ttk.Scrollbar(frame, orient="vertical", command=lb.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=lb.xview)
        lb.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        lb.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        return lb

    def _fit_names(self, names):
        """Widen the left panel so the longest name fits (up to 60% of the window)."""
        if not names:
            return
        needed = max(self.list_font.measure(n) for n in names) + 70
        window_width = max(self.winfo_width(), 1000)  # guard against a not-yet-measured window
        limit = max(int(window_width * 0.6), 300)
        if needed > self.side.winfo_width():
            self.panes.paneconfigure(self.side, width=min(needed, limit))

    def _build_plot(self):
        right = ttk.Frame(self.panes, padding=(6, 12, 12, 12))
        self.panes.add(right, minsize=300, stretch="always")
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _show_placeholder(self, text):
        self.fig.clf()
        self.ax = self.fig.add_subplot()
        self.ax.axis("off")
        self.ax.text(0.5, 0.5, text, ha="center", va="center", color="gray", fontsize=12)
        self.canvas.draw_idle()

    def _set_status(self, text, error=False):
        self.status.configure(text=text, foreground="#b00020" if error else "")

    # ------------------------------------------------------------------ actions
    def load_branch(self):
        branch = self.branch_var.get().strip()
        base = self.base_var.get().strip()
        if not branch:
            self._set_status("Type a branch name first.", error=True)
            return
        self.load_btn.configure(state="disabled")
        self._set_status(f"Loading '{branch}' (fetching from remote)...")
        self._reset_file_and_columns()
        self.file_list.delete(0, "end")
        self.base_label.configure(text="Before (split from): -")

        def work():  # runs in a background thread so the window stays responsive
            try:
                comparison = resolve_comparison(branch, base)
                files = changed_csvs(comparison.base_rev, comparison.head_rev)
                self.after(0, self._branch_loaded, comparison, files)
            except Exception as exc:  # GitError or anything unexpected
                self.after(0, self._branch_failed, str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _branch_loaded(self, comparison, files):
        self.load_btn.configure(state="normal")
        self.comparison, self.files = comparison, files
        sha = comparison.base_rev[:10]
        commit = f"merge-base {sha}" if comparison.base_is_split_point else sha
        self.base_label.configure(text=f"Before (split from): {comparison.base_label} ({commit})")
        description = f"'{comparison.head_label}' vs '{comparison.base_label}'"
        if not files:
            self._set_status(f"No CSV files changed: {description}.")
            self._show_placeholder("No changed CSV files.")
            return
        labels = [file_label(*f) for f in files]
        for label in labels:
            self.file_list.insert("end", label)
        self._fit_names(labels)
        self._set_status(f"{len(files)} CSV file(s) changed: {description}.")
        self._show_placeholder("Choose a file.")

    def _branch_failed(self, message):
        self.load_btn.configure(state="normal")
        self._set_status(message, error=True)
        self._show_placeholder("Could not load the comparison.")

    def _reset_file_and_columns(self):
        self.df_before = self.df_after = None
        self.columns = []
        for lb in (self.x_list, self.y_list):
            lb.delete(0, "end")

    def on_file_selected(self, _event=None):
        sel = self.file_list.curselection()
        if not sel:
            return
        status, old, new = self.files[sel[0]]
        self._reset_file_and_columns()
        try:
            comparison = self.comparison
            self.df_before = read_csv(None if status == "A" else run_git("show", f"{comparison.base_rev}:{old}"))
            self.df_after = read_csv(None if status == "D" else run_git("show", f"{comparison.head_rev}:{new}"))
        except Exception as exc:
            self._set_status(f"Could not read {new}:\n{exc}", error=True)
            return
        self.file_name = new

        cols_before = list(self.df_before.columns) if self.df_before is not None else []
        cols_after = list(self.df_after.columns) if self.df_after is not None else []
        self.columns = list(dict.fromkeys(cols_after + cols_before))
        labels = []
        for c in self.columns:
            note = ""
            if self.df_before is not None and self.df_after is not None:
                if c not in cols_before:
                    note = "  (new in branch)"
                elif c not in cols_after:
                    note = "  (removed in branch)"
            labels.append(f"{c}{note}")
        for lb in (self.x_list, self.y_list):
            lb.insert("end", *labels)
        self._fit_names(labels)

        self._set_status(f"File: {new}\n{len(self.columns)} columns. Choose the X and Y axes.")
        self._show_placeholder("Choose the X and Y columns.")

    def update_plot(self):
        xs, ys = self.x_list.curselection(), self.y_list.curselection()
        if not xs or not ys:
            return
        x, y = self.columns[xs[0]], self.columns[ys[0]]

        self.fig.clf()
        ax = self.fig.add_subplot()
        notes = []
        curves = ((self.df_before, "Before", dict(color="tab:blue", linestyle="--")),
                  (self.df_after, "After", dict(color="tab:orange")))
        x_is_date = False
        for df, name, style in curves:
            if df is None:
                notes.append(f"{name}: file does not exist in this version")
                continue
            if x not in df.columns or y not in df.columns:
                notes.append(f"{name}: '{x}' or '{y}' not in this version")
                continue
            xs = as_plottable(df[x])
            x_is_date = x_is_date or pd.api.types.is_datetime64_any_dtype(xs)
            marker = "." if len(df) <= 300 else None
            try:
                ax.plot(xs, as_plottable(df[y]), marker=marker, label=name, **style)
            except Exception as exc:
                notes.append(f"{name}: could not plot ({exc})")

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        # Title: CSV file name only (the full path is shown in the status text).
        ax.set_title(textwrap.fill(os.path.basename(self.file_name), 45), fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        if x_is_date:
            self.fig.autofmt_xdate()
        self.fig.tight_layout()
        self.canvas.draw_idle()

        text = f"File: {self.file_name}\nX: {x}\nY: {y}"
        if notes:
            text += "\nNote: " + "; ".join(notes)
        self._set_status(text)


def main():
    # Make sure we're inside a git repository; otherwise let the user pick one.
    try:
        git_text("rev-parse", "--show-toplevel")
    except GitError:
        picker = tk.Tk()
        picker.withdraw()
        folder = filedialog.askdirectory(title="Choose your local repository folder")
        if not folder:
            return
        os.chdir(folder)
        try:
            git_text("rev-parse", "--show-toplevel")
        except GitError:
            messagebox.showerror("CSV branch compare", f"'{folder}' is not a git repository.")
            return
        finally:
            picker.destroy()
    App().mainloop()


if __name__ == "__main__":
    main()
