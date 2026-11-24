#!/usr/bin/env python3
"""
csv_branch_compare.py - compare, as plots, the CSV files changed in a git branch or in
the working tree.

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
commit instead (e.g. my-branch~1 to see only the last commit).

Tick "Use current working tree" to compare the uncommitted files on disk (including new
untracked CSVs) instead of a branch; "before" is then HEAD unless "Base" is filled in,
and nothing is fetched.

"Summary (all files)" lists, for each changed file, how much its numbers changed (mean
and maximum difference, as a percentage of each column's size); double-click a file
there to see the same numbers per column, and on a column to see its values row by row.

Requires: git on PATH, Python 3.10+, pandas, matplotlib (Tkinter ships with Python)
"""
import io
import os
import subprocess
import sys
import textwrap
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

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
    head_rev: str | None
    """Commit of the 'after' side, or None to use the working tree (uncommitted files)."""
    head_label: str


def resolve_typed_base(base: str) -> str:
    """Commit of a base typed by the user, failing with a readable message."""
    if not ref_exists(base):
        raise GitError(f"Base '{base}' is not a known branch, tag or commit.")
    return git_text("rev-parse", "--verify", f"{base}^{{commit}}")


def resolve_comparison(branch: str, base: str, use_working_tree: bool) -> Comparison:
    """
    Work out 'before' and 'after' from what the user typed in the window.

    An empty `base` means the point where the branch split off the branch it was created
    from, or HEAD when comparing the working tree. The branch is resolved (and fetched)
    first, so a typed remote base is current; the working tree never fetches.
    """
    if use_working_tree:
        base = base or "HEAD"
        return Comparison(resolve_typed_base(base), base, False, None, "working tree (uncommitted)")

    if not branch:
        raise GitError("Type a branch name, or tick 'Use current working tree'.")
    head_ref, branch_name, remote = resolve_branch(branch)
    if base:
        return Comparison(resolve_typed_base(base), base, False, head_ref, branch)
    base_ref, base_label = find_base(head_ref, branch_name, remote)
    base_rev = git_text("merge-base", base_ref, head_ref)
    return Comparison(base_rev, base_label, True, head_ref, branch)


def changed_csvs(base_rev, head_rev):
    """Return (status, old_path, new_path) for every CSV changed between 'before' and 'after'.
    A head_rev of None means the working tree, which also includes new untracked CSVs.
    Paths are relative to the repository root."""
    diff_args = ["diff", "--name-status", "-M", "-z", base_rev]
    if head_rev is not None:
        diff_args.append(head_rev)
    raw = run_git(*diff_args)
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
    if head_rev is None:
        # ":/" and --full-name keep paths repo-relative even when run from a subfolder.
        untracked = run_git("ls-files", "--others", "--exclude-standard", "--full-name", "-z", ":/")
        for path in untracked.decode("utf-8", "replace").split("\0"):
            if path.lower().endswith(".csv"):
                files.append(("A", path, path))
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


def read_version(rev: str | None, path: str, repo_root: str) -> bytes:
    """Bytes of a file at a commit, or from disk when rev is None (working tree)."""
    if rev is None:
        return (Path(repo_root) / path).read_bytes()
    return run_git("show", f"{rev}:{path}")


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


def read_pair(comparison: Comparison, repo_root: str, status: str, old: str, new: str):
    """Return the (before, after) DataFrames of a changed file (None where it does not exist)."""
    before = None if status == "A" else read_csv(read_version(comparison.base_rev, old, repo_root))
    after = None if status == "D" else read_csv(read_version(comparison.head_rev, new, repo_root))
    return before, after


@dataclass(frozen=True)
class ColumnChange:
    """How much one numeric column changed between 'before' and 'after'."""

    column: str
    mean_difference: float
    """Mean of |after - before|, in the column's own unit."""
    max_difference: float
    mean_percent: float
    """Mean difference as a percentage of the column's size (its largest absolute value)."""
    max_percent: float


def column_changes(before: pd.DataFrame, after: pd.DataFrame) -> list[ColumnChange]:
    """
    Compare the numeric columns present on both sides, row by row.

    Percentages are relative to each column's largest absolute value, so values close to
    zero do not blow them up. Only the rows present on both sides are compared.
    """
    row_count = min(len(before), len(after))
    changes = []
    for column in after.columns:
        if column not in before.columns:
            continue
        old = pd.to_numeric(before[column].iloc[:row_count], errors="coerce")
        new = pd.to_numeric(after[column].iloc[:row_count], errors="coerce")
        valid = old.notna() & new.notna()
        if not valid.any():
            # Keep columns that are numeric but empty on a side, so their values can be
            # inspected; skip text columns.
            both_empty = before[column].iloc[:row_count].isna().all() and after[column].iloc[:row_count].isna().all()
            if old.notna().any() or new.notna().any() or both_empty:
                nan = float("nan")
                changes.append(ColumnChange(str(column), nan, nan, nan, nan))
            continue
        difference = (new[valid] - old[valid]).abs()
        size = max(old[valid].abs().max(), new[valid].abs().max())
        scale = 100.0 / size if size > 0 else 0.0
        changes.append(ColumnChange(
            column=str(column),
            mean_difference=float(difference.mean()),
            max_difference=float(difference.max()),
            mean_percent=float(difference.mean() * scale),
            max_percent=float(difference.max() * scale),
        ))
    return changes


@dataclass(frozen=True)
class FileChange:
    """Summary of how much one CSV file changed (one line of the summary table)."""

    label: str
    status: str
    old: str
    new: str
    rows_before: int | None
    rows_after: int | None
    changed_columns: int
    numeric_columns: int
    mean_percent: float | None
    """Mean of the per-column mean percentages, over the columns that changed."""
    max_percent: float | None
    worst_column: str
    columns: list[ColumnChange]

    @classmethod
    def FromFrames(cls, status: str, old: str, new: str,
                   before: pd.DataFrame | None, after: pd.DataFrame | None) -> "FileChange":
        """Summarize a file; added or deleted files have no numbers to compare."""
        label = file_label(status, old, new)
        rows_before = None if before is None else len(before)
        rows_after = None if after is None else len(after)
        if before is None or after is None:
            return cls(label, status, old, new, rows_before, rows_after, 0, 0, None, None, "", [])
        columns = column_changes(before, after)
        changed = [c for c in columns if c.max_difference > 0]
        if not changed:
            return cls(label, status, old, new, rows_before, rows_after, 0, len(columns), 0.0, 0.0, "", columns)
        worst = max(changed, key=lambda c: c.max_percent)
        return cls(
            label, status, old, new, rows_before, rows_after, len(changed), len(columns),
            sum(c.mean_percent for c in changed) / len(changed), worst.max_percent, worst.column, columns,
        )


# --------------------------------------------------------------------------- tables
class TableWindow(tk.Toplevel):
    """
    A sortable table. Each column is (header, kind) with kind 'text', 'number' or
    'percent'; rows hold raw values (None is shown as '-'). Click a header to sort.

    Copying (Ctrl+C, right-click, buttons) yields tab-separated raw values, so it pastes
    into Excel at full precision.
    """

    def __init__(self, master, title, columns, rows, on_open=None, geometry="1100x500"):
        super().__init__(master)
        self.title(title)
        self.geometry(geometry)
        self.columns = columns
        self.rows = rows
        self.on_open = on_open
        self.sort_column = None
        self.sort_reverse = True

        frame = ttk.Frame(self, padding=6)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(frame, columns=[h for h, _ in columns], show="headings")
        ys = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        for index, (header, kind) in enumerate(columns):
            self.tree.heading(header, text=header, command=lambda i=index: self._sort_by(i))
            width = 420 if index == 0 else 110
            self.tree.column(header, width=width, anchor="w" if kind == "text" else "e", stretch=index == 0)
        if on_open is not None:
            self.tree.bind("<Double-1>", self._open_selected)

        self.tree.bind("<Control-c>", lambda e: self._copy_rows(self.tree.selection()))
        self.tree.bind("<Control-C>", lambda e: self._copy_rows(self.tree.selection()))
        self.tree.bind("<Control-a>", self._select_all)
        self.tree.bind("<Control-A>", self._select_all)
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Copy selected rows",
                              command=lambda: self._copy_rows(self.tree.selection()))
        self.menu.add_command(label="Copy all rows (with header)", command=self._copy_all)
        self.tree.bind("<Button-3>", self._show_menu)

        footer = ttk.Frame(frame)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(footer, text="Copy all (with header)", command=self._copy_all).pack(side="left")
        ttk.Button(footer, text="Copy selected", command=lambda: self._copy_rows(self.tree.selection())
                   ).pack(side="left", padx=(6, 0))
        hint = "Ctrl+C copies the selected rows; click a header to sort."
        if on_open is not None:
            hint += " Double-click a row to see its details."
        ttk.Label(footer, text=hint, foreground="gray").pack(side="left", padx=(10, 0))
        self._fill()

    @staticmethod
    def _format(value, kind):
        if value is None:
            return "-"
        if kind == "percent":
            return f"{value:.3f} %"
        if kind == "number":
            return f"{value:.4g}"
        return str(value)

    def _fill(self):
        self.tree.delete(*self.tree.get_children())
        for index, row in enumerate(self.rows):
            shown = [self._format(v, kind) for v, (_, kind) in zip(row, self.columns)]
            self.tree.insert("", "end", iid=str(index), values=shown)

    def _sort_by(self, column_index):
        """Sort by a column, toggling the direction on repeated clicks; missing values go last."""
        self.sort_reverse = not self.sort_reverse if self.sort_column == column_index else True
        self.sort_column = column_index

        def is_missing(value):
            return value is None or (isinstance(value, float) and value != value)  # None or NaN

        present = [r for r in self.rows if not is_missing(r[column_index])]
        missing = [r for r in self.rows if is_missing(r[column_index])]
        present.sort(key=lambda r: r[column_index], reverse=self.sort_reverse)
        self.rows = present + missing
        self._fill()

    def _select_all(self, _event=None):
        self.tree.selection_set(self.tree.get_children())
        return "break"

    def _show_menu(self, event):
        row = self.tree.identify_row(event.y)
        if row and row not in self.tree.selection():
            self.tree.selection_set(row)
        self.menu.tk_popup(event.x_root, event.y_root)

    @staticmethod
    def _plain(value):
        return "" if value is None else str(value)

    def _copy_text(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()  # Keeps the text on the clipboard after the window is closed.

    def _copy_rows(self, selection):
        """Copy the selected rows (all rows if none is selected), without the header."""
        indices = [int(iid) for iid in selection] if selection else range(len(self.rows))
        lines = ["\t".join(self._plain(v) for v in self.rows[i]) for i in sorted(indices)]
        self._copy_text("\n".join(lines))
        return "break"

    def _copy_all(self):
        header = "\t".join(h for h, _ in self.columns)
        lines = ["\t".join(self._plain(v) for v in row) for row in self.rows]
        self._copy_text("\n".join([header, *lines]))

    def _open_selected(self, _event):
        selection = self.tree.selection()
        if selection:
            self.on_open(self.rows[int(selection[0])])


# --------------------------------------------------------------------------- GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV branch compare")
        self.geometry("1150x720")
        self.minsize(800, 500)

        self.comparison = None
        self.repo_root = git_text("rev-parse", "--show-toplevel")
        self.files = []
        self.df_before = self.df_after = None
        self.file_name = ""
        self.columns = []
        self.file_changes = None  # Filled by "Summary (all files)", reset on every load.

        self._build_controls()
        self._build_plot()
        self._show_placeholder("Enter a branch name (or tick 'Use current working tree') and press Load.")

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
        self.branch_entry = ttk.Entry(row, textvariable=self.branch_var)
        self.branch_entry.pack(side="left", fill="x", expand=True)
        self.branch_entry.bind("<Return>", lambda e: self.load_branch())
        self.branch_entry.focus_set()
        self.load_btn = ttk.Button(row, text="Load", command=self.load_branch)
        self.load_btn.pack(side="left", padx=(6, 0))
        ttk.Label(side, text="my-branch = local, origin/my-branch = remote",
                  foreground="gray").grid(row=2, column=0, sticky="w")

        self.working_tree_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Use current working tree (including uncommitted changes) as 'after'",
                        variable=self.working_tree_var,
                        command=self._on_working_tree_toggled).grid(row=3, column=0, sticky="w", pady=(6, 0))

        ttk.Label(side, text="Base (before) - optional").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.base_var = tk.StringVar()
        base_entry = ttk.Entry(side, textvariable=self.base_var)
        base_entry.grid(row=5, column=0, sticky="ew", pady=(2, 2))
        base_entry.bind("<Return>", lambda e: self.load_branch())
        ttk.Label(side, text="empty = the branch it was created from (HEAD for the working tree)",
                  foreground="gray").grid(row=6, column=0, sticky="w")

        self.base_label = ttk.Label(side, text="Before (split from): -")
        self.base_label.grid(row=7, column=0, sticky="w", pady=(8, 0))

        self.status = ttk.Label(side, text="", justify="left")
        self.status.grid(row=8, column=0, sticky="ew", pady=(10, 6))
        side.bind("<Configure>", lambda e: self.status.configure(wraplength=max(e.width - 30, 100)))

        self.summary_btn = ttk.Button(side, text="Summary (all files)", command=self.show_summary,
                                      state="disabled")
        self.summary_btn.grid(row=9, column=0, sticky="ew", pady=(0, 4))

        self.file_list = self._make_list(side, "Changed CSV files", first_row=10, weight=2)
        self.x_list = self._make_list(side, "X axis", first_row=12, weight=1)
        self.y_list = self._make_list(side, "Y axis", first_row=14, weight=1)
        self.file_list.bind("<<ListboxSelect>>", self.on_file_selected)
        for lb in (self.x_list, self.y_list):
            lb.bind("<<ListboxSelect>>", lambda e: self.update_plot())

        self.list_font = tkfont.Font(font=self.file_list.cget("font"))

    def _on_working_tree_toggled(self):
        self.branch_entry.configure(state="disabled" if self.working_tree_var.get() else "normal")

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
        use_working_tree = self.working_tree_var.get()
        if not branch and not use_working_tree:
            self._set_status("Type a branch name first, or tick 'Use current working tree'.", error=True)
            return
        self.load_btn.configure(state="disabled")
        self.summary_btn.configure(state="disabled")
        self._set_status("Loading..." if use_working_tree else f"Loading '{branch}' (fetching from remote)...")
        self._reset_file_and_columns()
        self.file_changes = None
        self.file_list.delete(0, "end")
        self.base_label.configure(text="Before (split from): -")

        def work():  # runs in a background thread so the window stays responsive
            try:
                comparison = resolve_comparison(branch, base, use_working_tree)
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
        self.summary_btn.configure(state="normal")
        self._set_status(f"{len(files)} CSV file(s) changed: {description}.")
        self._show_placeholder("Choose a file, or open the summary.")

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
            self.df_before, self.df_after = read_pair(self.comparison, self.repo_root, status, old, new)
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
                    note = "  (new in 'after')"
                elif c not in cols_after:
                    note = "  (removed in 'after')"
            labels.append(f"{c}{note}")
        for lb in (self.x_list, self.y_list):
            lb.insert("end", *labels)
        self._fit_names(labels)

        self._set_status(f"File: {new}\n{len(self.columns)} columns. Choose the X and Y axes.")
        self._show_placeholder("Choose the X and Y columns.")

    # ------------------------------------------------------------------ summary
    def show_summary(self):
        if self.file_changes is not None:
            self._open_summary()
            return
        self.summary_btn.configure(state="disabled")
        self._set_status(f"Reading {len(self.files)} file(s) for the summary...")
        comparison, repo_root, files = self.comparison, self.repo_root, list(self.files)

        def work():
            try:
                changes = [FileChange.FromFrames(status, old, new,
                                                 *read_pair(comparison, repo_root, status, old, new))
                           for status, old, new in files]
                self.after(0, self._summary_ready, comparison, changes)
            except Exception as exc:
                self.after(0, self._summary_failed, comparison, str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _summary_failed(self, comparison, message):
        if comparison is not self.comparison:  # Another load started meanwhile.
            return
        self.summary_btn.configure(state="normal")
        self._set_status(f"Could not build the summary:\n{message}", error=True)

    def _summary_ready(self, comparison, changes):
        if comparison is not self.comparison:
            return
        self.file_changes = changes
        self.summary_btn.configure(state="normal")
        self._set_status(f"Summary of {len(changes)} file(s) ready.")
        self._open_summary()

    def _open_summary(self):
        columns = [("File", "text"), ("Rows before", "number"), ("Rows after", "number"),
                   ("Columns changed", "number"), ("Numeric columns", "number"),
                   ("Mean difference (%)", "percent"), ("Max difference (%)", "percent"),
                   ("Column with the max difference", "text")]
        rows = [(c.label, c.rows_before, c.rows_after, c.changed_columns, c.numeric_columns,
                 c.mean_percent, c.max_percent, c.worst_column) for c in self.file_changes]
        rows.sort(key=lambda r: (r[6] is None, -(r[6] or 0.0)))  # Biggest change first.
        by_label = {c.label: c for c in self.file_changes}
        title = f"Summary: '{self.comparison.head_label}' vs '{self.comparison.base_label}'"
        TableWindow(self, title, columns, rows,
                    on_open=lambda row: self._open_columns(by_label[row[0]]), geometry="1250x520")

    def _open_columns(self, file_change):
        if not file_change.columns:
            messagebox.showinfo("CSV branch compare",
                                "No numeric columns to compare (file added, deleted or non-numeric).")
            return
        columns = [("Column", "text"), ("Mean difference", "number"), ("Max difference", "number"),
                   ("Mean difference (%)", "percent"), ("Max difference (%)", "percent")]
        rows = [(c.column, c.mean_difference, c.max_difference, c.mean_percent, c.max_percent)
                for c in file_change.columns]
        rows.sort(key=lambda r: (r[4] != r[4], -r[4] if r[4] == r[4] else 0.0))  # NaN last.
        note = " (differences are in each column's own unit; % is relative to the column's size)"
        TableWindow(self, file_change.label + note, columns, rows,
                    on_open=lambda row: self._open_values(file_change, row[0]))

    def _open_values(self, file_change, column):
        """
        Show one column's values, before and after, row by row.

        The file's first other column is shown alongside as a row label, which is usually
        the X/time column.
        """
        try:
            before, after = read_pair(self.comparison, self.repo_root,
                                      file_change.status, file_change.old, file_change.new)
        except Exception as exc:
            messagebox.showerror("CSV branch compare", f"Could not read the file:\n{exc}")
            return
        frames = {"before": before, "after": after}
        series = {name: pd.to_numeric(frame[column], errors="coerce").tolist()
                  for name, frame in frames.items() if frame is not None and column in frame.columns}
        reference = after if after is not None else before
        label_column = next((c for c in reference.columns if str(c) != column), None)
        labels = {}
        if label_column is not None:
            labels = {name: frame[label_column].tolist()
                      for name, frame in frames.items() if frame is not None and label_column in frame.columns}
        total = max((len(v) for v in series.values()), default=0)

        def at(values, index):
            return values[index] if values is not None and index < len(values) else None

        rows = []
        for index in range(total):
            old, new = at(series.get("before"), index), at(series.get("after"), index)
            difference = new - old if old is not None and new is not None else None
            label = at(labels.get("after"), index)
            if label is None:
                label = at(labels.get("before"), index)
            rows.append((index, "" if label is None else label, old, new, difference))
        columns = [("Row", "number"), (str(label_column) if label_column is not None else "-", "text"),
                   ("Before", "number"), ("After", "number"), ("After - before", "number")]
        title = f"{column}  |  {os.path.basename(file_change.new)}"
        TableWindow(self, title, columns, rows, geometry="900x600")

    # ------------------------------------------------------------------ plot
    def update_plot(self):
        xs, ys = self.x_list.curselection(), self.y_list.curselection()
        if not xs or not ys:
            return
        x, y = self.columns[xs[0]], self.columns[ys[0]]

        self.fig.clf()
        ax = self.fig.add_subplot()
        notes = []
        curves = ((self.df_before, "before", dict(color="tab:blue", linestyle="--")),
                  (self.df_after, "after", dict(color="tab:orange")))
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
                ax.plot(xs, as_plottable(df[y]), marker=marker, label=f"{name} ({len(df)} rows)", **style)
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
