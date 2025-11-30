import math
import heapq
import threading
import random
import tkinter as tk
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Optional


def load_word_list(path: Path) -> List[str]:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            raw = [line.strip().lower() for line in fh if line.strip()]
        filtered = [w for w in raw if len(w) == 5 and w.isalpha()]
        seen = set()
        out = []
        for w in filtered:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out

    return [
        "apple","beach","brain","bread","brush","chair","chest","chord","click","clock",
        "cloud","dance","diary","drink","drive","earth","feast","field","fruit","glass",
        "grape","green","ghost","heart","house","juice","light","lemon","melon","money",
        "music","night","ocean","party","piano","pilot","plane","phone","pizza","power",
        "radio","river","robot","shirt","shoes","smile","snake","space","spoon","storm",
        "table","tiger","toast","touch","train","truck","voice","water","watch","whale",
        "world","write","youth","zebra","lunch"
    ]


DEFAULT_WORDLIST_PATH = Path(__file__).parent / "wordlists" / "wordle_words.txt"
WORD_LIST = load_word_list(DEFAULT_WORDLIST_PATH)



BG = "#121213"               # page background
EMPTY_BG = BG                # empty tile background (same as page)
EMPTY_BORDER = "#3a3a3c"     # empty tile border
EMPTY_TEXT = "#d7dadc"       # light gray text on empty tiles
KEY_BG = "#818384"           # keyboard key background (light gray)
KEY_ACTIVE_BG = "#6e6f70"    # keyboard key active background
COLOR_CORRECT = "#6aaa64"    # green
COLOR_PRESENT = "#c9b458"    # yellow
COLOR_ABSENT = "#3a3a3c"     # dark gray for absent
COLOR_TEXT_FILLED = "#ffffff"  # white text on colored tiles

ROWS = 6
COLS = 5
CELL_SIZE = 74
REVEAL_DELAY_MS = 260


def get_pattern(guess: str, target: str) -> Tuple[int, ...]:
    pattern = [0] * 5
    target_counts = Counter(target)
    used = [False] * 5

    # Mark greens first
    for i in range(5):
        if guess[i] == target[i]:
            pattern[i] = 2
            target_counts[guess[i]] -= 1
            used[i] = True

    # Mark yellows
    for i in range(5):
        if not used[i] and guess[i] in target_counts and target_counts[guess[i]] > 0:
            pattern[i] = 1
            target_counts[guess[i]] -= 1

    return tuple(pattern)


def filter_words(words: List[str], guess: str, pattern: Tuple[int, ...]) -> List[str]:
    return [word for word in words if get_pattern(guess, word) == pattern]


def calculate_entropy(guess: str, candidates: List[str]) -> float:
    pattern_counts = Counter()
    for word in candidates:
        pattern = get_pattern(guess, word)
        pattern_counts[pattern] += 1
    total = len(candidates)
    entropy = 0.0
    for count in pattern_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def heuristic_remaining_candidates(candidates: List[str], known_constraints: dict) -> float:
    n = len(candidates)
    if n == 0:
        return float('inf')
    if n == 1:
        return 1.0
    h = math.log2(n)
    if n <= 4:
        h = 1.0
    elif n <= 10:
        h = 1.5
    return h


class SearchNode:
    def __init__(self, candidates: List[str], guess_history: List[Tuple[str, Tuple[int, ...]]],
                 g_cost: float, known_constraints: dict = None):
        self.candidates = candidates
        self.guess_history = guess_history
        self.g_cost = g_cost
        self.known_constraints = known_constraints or {}
        self.h_cost = heuristic_remaining_candidates(candidates, self.known_constraints)
        self.f_cost = self.g_cost + self.h_cost

    def __lt__(self, other):
        if self.f_cost != other.f_cost:
            return self.f_cost < other.f_cost
        return len(self.candidates) < len(other.candidates)


def find_best_guess_astar(candidates: List[str], all_words: List[str] = None) -> str:
    if not candidates:
        raise ValueError("No candidates available to select a guess.")
    if len(candidates) <= 2:
        return candidates[0]
    if all_words is None:
        all_words = candidates
    seed_guesses = candidates[:50] + ['slate', 'crate', 'trace', 'arose', 'about']
    seen = set()
    guess_pool = []
    for g in seed_guesses:
        if g in all_words and g not in seen:
            seen.add(g)
            guess_pool.append(g)
    if not guess_pool:
        guess_pool = list(all_words[:50]) if all_words else candidates[:1]
    best_guess = guess_pool[0]
    best_score = -float('inf')
    for guess in guess_pool:
        entropy = calculate_entropy(guess, candidates)
        in_list_bonus = 0.2 if guess in candidates else 0
        score = entropy + in_list_bonus
        if score > best_score:
            best_score = score
            best_guess = guess
    return best_guess


def solve_wordle_astar(target_word: str, all_words: List[str], max_guesses: int = 6,
                       verbose: bool = False) -> Tuple[List[Tuple[str, Tuple[int, ...]]], int]:
    """
    True A*: expand all feedback branches for chosen guesses.
    Returns: (guess_history as list of (guess, pattern), number_of_expanded_nodes)
    """
    if not all_words:
        raise ValueError("all_words must be a non-empty list.")

    initial_node = SearchNode(candidates=all_words.copy(), guess_history=[], g_cost=0.0)
    frontier = []
    heapq.heappush(frontier, initial_node)
    expanded_nodes = 0
    closed = {}

    while frontier and expanded_nodes < 10000:
        current = heapq.heappop(frontier)
        expanded_nodes += 1

        # Goal test: single candidate equal to target
        if len(current.candidates) == 1 and current.candidates[0] == target_word:
            return current.guess_history, expanded_nodes

        key = tuple(sorted(current.candidates))
        prev_g = closed.get(key)
        if prev_g is not None and current.g_cost >= prev_g:
            continue
        closed[key] = current.g_cost

        guess = find_best_guess_astar(current.candidates, all_words)

        # Group candidates by pattern for this guess
        pattern_groups = {}
        for cand in current.candidates:
            pat = get_pattern(guess, cand)
            pattern_groups.setdefault(pat, []).append(cand)

        for pat, group in pattern_groups.items():
            new_candidates = group
            if not new_candidates:
                continue
            new_history = current.guess_history + [(guess, pat)]
            new_g = current.g_cost + 1.0
            child = SearchNode(candidates=new_candidates, guess_history=new_history, g_cost=new_g,
                               known_constraints=current.known_constraints.copy())
            child_key = tuple(sorted(child.candidates))
            if child_key in closed and closed[child_key] <= child.g_cost:
                continue
            heapq.heappush(frontier, child)

    return [], expanded_nodes


# -----------------------------
# UI: WordleGame with Auto-Solve
# -----------------------------
class WordleGame:

    def __init__(self, root: tk.Tk, debug: bool = False) -> None:
        self.root = root
        self.debug = debug

        self.root.title("Wordle")
        self.root.geometry("660x940")
        self.root.configure(bg=BG)

        self.target_word: str = ""
        self.current_guess_num: int = 0
        self.current_guess_str: str = ""
        self.game_over: bool = False
        self.revealing: bool = False

        # UI state
        self.cells: List[List[tuple]] = []
        self.message_label: Optional[tk.Label] = None
        self.key_buttons: dict = {}
        self._flash_after_id: Optional[str] = None

        self.setup_ui()
        self.start_new_game()

    def setup_ui(self) -> None:
        # Title
        title_label = tk.Label(self.root, text="WORDLE", font=("Helvetica", 40, "bold"), bg=BG, fg="white")
        title_label.pack(pady=(20, 4))

        # message / status
        self.message_label = tk.Label(self.root, text="", font=("Helvetica", 12), bg=BG, fg=EMPTY_TEXT)
        self.message_label.pack(pady=(0, 6))

        # Grid container (centered)
        self.grid_frame = tk.Frame(self.root, bg=BG)
        self.grid_frame.pack(pady=(6, 18))

        # Build grid
        self.cells = []
        for row in range(ROWS):
            row_cells = []
            for col in range(COLS):
                cell_frame = tk.Frame(self.grid_frame, width=CELL_SIZE, height=CELL_SIZE, bg=EMPTY_BG,
                                      highlightbackground=EMPTY_BORDER, highlightthickness=2)
                cell_frame.grid(row=row, column=col, padx=6, pady=6)
                cell_frame.pack_propagate(False)
                lbl = tk.Label(cell_frame, text="", font=("Helvetica", 36, "bold"), bg=EMPTY_BG, fg=EMPTY_TEXT)
                lbl.pack(expand=True, fill="both")
                row_cells.append((cell_frame, lbl))
            self.cells.append(row_cells)

        # On-screen keyboard
        kb_frame = tk.Frame(self.root, bg=BG)
        kb_frame.pack(pady=(6, 12))

        rows = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"]
        for r_idx, keys in enumerate(rows):
            row_frame = tk.Frame(kb_frame, bg=BG)
            row_frame.pack(pady=6)
            if r_idx == 2:
                enter_btn = tk.Button(row_frame, text="ENTER", width=6, height=2, bg=KEY_BG, fg="white",
                                      activebackground=KEY_ACTIVE_BG, font=("Helvetica", 10, "bold"),
                                      command=lambda: self.on_screen_key("ENTER"))
                enter_btn.pack(side="left", padx=6)
            for ch in keys:
                btn = tk.Button(row_frame, text=ch, width=4, height=2, bg=KEY_BG, fg="white",
                                activebackground=KEY_ACTIVE_BG, font=("Helvetica", 12, "bold"),
                                command=lambda c=ch: self.on_screen_key(c))
                btn.pack(side="left", padx=6)
                self.key_buttons[ch] = btn
            if r_idx == 2:
                back_btn = tk.Button(row_frame, text="⌫", width=6, height=2, bg=KEY_BG, fg="white",
                                     activebackground=KEY_ACTIVE_BG, font=("Helvetica", 12, "bold"),
                                     command=lambda: self.on_screen_key("BACK"))
                back_btn.pack(side="left", padx=6)
                self.key_buttons["BACK"] = back_btn

        # Buttons: New Game and Auto Solve
        controls_frame = tk.Frame(self.root, bg=BG)
        controls_frame.pack(pady=(6, 12))

        restart_btn = tk.Button(controls_frame, text="New Game", command=self.start_new_game,
                                font=("Helvetica", 11, "bold"), bg="#3a3a3c", fg="white",
                                activebackground="#4a4a4a", relief="flat")
        restart_btn.pack(side="left", padx=8)

        auto_btn = tk.Button(controls_frame, text="Auto Solve", command=self.auto_solve,
                             font=("Helvetica", 11, "bold"), bg="#1f6feb", fg="white",
                             activebackground="#155bb5", relief="flat")
        auto_btn.pack(side="left", padx=8)

        # Bind keyboard input
        self.root.bind("<Key>", self.handle_keypress)
        self.root.focus_set()

    def start_new_game(self) -> None:
        self.target_word = random.choice(WORD_LIST)
        self.current_guess_num = 0
        self.current_guess_str = ""
        self.game_over = False
        self.revealing = False
        if self.message_label:
            self.message_label.config(text="", fg=EMPTY_TEXT)

        # Reset grid to empty dark tiles with subtle border
        for row in range(ROWS):
            for col in range(COLS):
                frame, lbl = self.cells[row][col]
                lbl.config(text="", bg=EMPTY_BG, fg=EMPTY_TEXT)
                frame.config(bg=EMPTY_BG, highlightbackground=EMPTY_BORDER)

        # Reset keyboard colors
        for k, btn in self.key_buttons.items():
            btn.config(bg=KEY_BG, fg="white")

        if self.debug:
            print(f"DEBUG: target = {self.target_word}")

        self.root.focus_set()

    def handle_keypress(self, event: tk.Event) -> None:
        if self.game_over or self.revealing:
            return
        char = (event.char or "").upper()
        keysym = (event.keysym or "").upper()
        if len(char) == 1 and char.isalpha():
            self.on_letter(char)
            return
        if keysym == "BACKSPACE":
            self.on_backspace()
            return
        if keysym in ("RETURN", "ENTER"):
            self.on_enter()
            return

    def on_screen_key(self, key: str) -> None:
        if self.game_over or self.revealing:
            return
        if key == "ENTER":
            self.on_enter()
        elif key == "BACK":
            self.on_backspace()
        else:
            self.on_letter(key)

    def on_letter(self, ch: str) -> None:
        if len(self.current_guess_str) < COLS:
            self.current_guess_str += ch.upper()
            self.update_active_row()

    def on_backspace(self) -> None:
        if len(self.current_guess_str) > 0:
            self.current_guess_str = self.current_guess_str[:-1]
            self.update_active_row()

    def on_enter(self) -> None:
        if len(self.current_guess_str) != COLS:
            self.flash_message("Not enough letters")
            return
        if self.current_guess_str.lower() not in WORD_LIST:
            self.flash_message("Not in word list")
            return
        self.reveal_current_guess()

    def update_active_row(self) -> None:
        row = self.current_guess_num
        for col in range(COLS):
            frame, lbl = self.cells[row][col]
            if col < len(self.current_guess_str):
                lbl.config(text=self.current_guess_str[col], fg=EMPTY_TEXT, bg=EMPTY_BG)
                frame.config(highlightbackground=EMPTY_BORDER)
            else:
                lbl.config(text="", fg=EMPTY_TEXT, bg=EMPTY_BG)
                frame.config(highlightbackground=EMPTY_BORDER)

    def reveal_current_guess(self) -> None:
        """Sequential reveal animation like NYT Wordle (simple illusion)."""
        self.revealing = True
        guess = self.current_guess_str.upper()
        target = self.target_word.upper()

        # Determine results (greens then yellows)
        result_colors = [COLOR_ABSENT] * COLS
        t_list = list(target)
        g_list = list(guess)

        for i in range(COLS):
            if g_list[i] == t_list[i]:
                result_colors[i] = COLOR_CORRECT
                t_list[i] = None
                g_list[i] = None

        for i in range(COLS):
            if g_list[i] is not None and g_list[i] in t_list:
                result_colors[i] = COLOR_PRESENT
                t_list[t_list.index(g_list[i])] = None

        # animate tiles with delays
        for i in range(COLS):
            delay = i * REVEAL_DELAY_MS
            self.root.after(delay, self._make_reveal_closure(i, guess[i], result_colors[i]))

        total_time = (COLS - 1) * REVEAL_DELAY_MS + 300
        self.root.after(total_time, self._after_reveal_actions)

    def _make_reveal_closure(self, index: int, ch: str, color: str):
        def _reveal():
            frame, lbl = self.cells[self.current_guess_num][index]
            # quick "flip" illusion: blank -> colored tile with letter
            lbl.config(text="", bg=EMPTY_BG)
            frame.config(bg=EMPTY_BG, highlightbackground=EMPTY_BORDER)
            def _show_colored():
                lbl.config(text=ch, bg=color, fg=COLOR_TEXT_FILLED)
                frame.config(bg=color, highlightbackground=color)
                # update on-screen keyboard color (never downgrade green)
                btn = self.key_buttons.get(ch)
                if btn:
                    current = btn.cget("bg")
                    if current != COLOR_CORRECT:
                        if color == COLOR_CORRECT:
                            btn.config(bg=COLOR_CORRECT)
                        elif color == COLOR_PRESENT and current not in (COLOR_CORRECT, COLOR_PRESENT):
                            btn.config(bg=COLOR_PRESENT)
                        elif color == COLOR_ABSENT and current not in (COLOR_CORRECT, COLOR_PRESENT):
                            btn.config(bg=COLOR_ABSENT)
            self.root.after(130, _show_colored)
        return _reveal

    def _after_reveal_actions(self) -> None:
        guess = self.current_guess_str.upper()
        target = self.target_word.upper()

        if guess == target:
            self.game_over = True
            if self.message_label:
                self.message_label.config(text="SPLENDID!", fg=COLOR_CORRECT)
            self.revealing = False
            return

        self.current_guess_num += 1
        self.current_guess_str = ""
        self.revealing = False

        if self.current_guess_num >= ROWS:
            self.game_over = True
            if self.message_label:
                self.message_label.config(text=f"Valiant effort! Word: {target}", fg=EMPTY_TEXT)

    def flash_message(self, msg: str, duration: int = 1000) -> None:
        if not self.message_label:
            return
        if self._flash_after_id:
            try:
                self.root.after_cancel(self._flash_after_id)
            except Exception:
                pass
            self._flash_after_id = None

        original = self.message_label.cget("text")
        original_fg = self.message_label.cget("fg")
        self.message_label.config(text=msg, fg="white")

        def _restore():
            self.message_label.config(text=original, fg=original_fg)
            self._flash_after_id = None

        self._flash_after_id = self.root.after(duration, _restore)

    # -----------------------------
    # Auto-solve integration
    # -----------------------------
    def auto_solve(self) -> None:
        """Start auto-solve in a background thread; animation runs on main thread."""
        if self.revealing or self.game_over:
            return
        # disable interactions while solving
        self.flash_message("Auto-solving...", duration=1500)
        solver_thread = threading.Thread(target=self._run_solver_thread, daemon=True)
        solver_thread.start()

    def _run_solver_thread(self) -> None:
        # Use lowercase canonical form for comparisons
        target = self.target_word.lower()

        candidates = WORD_LIST.copy()
        history: List[Tuple[str, Tuple[int, ...]]] = []

        for _ in range(ROWS):
            if not candidates:
                break
            # choose best guess from remaining candidates (A*-informed)
            guess = find_best_guess_astar(candidates, WORD_LIST)
            pattern = get_pattern(guess, target)
            history.append((guess, pattern))
            if pattern == (2, 2, 2, 2, 2):
                break
            # filter candidates for next iteration
            candidates = filter_words(candidates, guess, pattern)

        # schedule animation of the solver's guess sequence on the main thread
        self.root.after(0, lambda h=history, t=target: self._animate_solver_history(h, t))

    def _animate_solver_history(self, history: List[Tuple[str, Tuple[int, ...]]], target_override: Optional[str] = None) -> None:
        if not history:
            self.flash_message("Solver failed", duration=1500)
            return

        # Reset UI to start solving from top of grid
        self.start_new_game()

        # Restore solver's target (start_new_game picked a new random target)
        if target_override:
            self.target_word = target_override

        delay_acc = 0
        for guess, pattern in history:
            # schedule setting current_guess_str and revealing
            def make_step(g=guess):
                # place guess in current row and reveal using existing reveal flow (it recomputes pattern)
                self.current_guess_str = g.upper()
                self.update_active_row()
                self.reveal_current_guess()
            self.root.after(delay_acc, make_step)
            # increment by reveal length
            delay_acc += (COLS - 1) * REVEAL_DELAY_MS + 500


if __name__ == "__main__":
    root = tk.Tk()
    game = WordleGame(root, debug=False)
    root.mainloop()