"""Long-run picross autoplay: many levels with lag-clearing app restarts.

The fill itself is reliable (per-run swipe), but the game's touch input slowly
lags over a long unattended run -- so we restart the app every `CHUNK` levels
(and whenever a level fails to complete), which clears the lag. Saves persist.

Navigation never uses the in-game RESET (it wipes progress) and never presses
BACK on the level map (that opens the app-exit dialog).

    python3 run100.py [target]      # logs to /tmp/run100.log
"""
import sys
import time

import auto
import clue_ocr
import play

PKG = "com.gmail.joystudio808.nonogram"
LOG = "/tmp/run100.log"
# The per-run swipe fill is reliable, so a healthy run doesn't need preventive
# restarts -- we restart only when a level actually fails (lag/corruption).
# Set CHUNK to a finite value to re-enable periodic restarts as insurance.
CHUNK = 0                        # 0 = restart only on failure, never on schedule


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def restart_app():
    log("restart app (clear input lag)")
    auto._sh("shell", "am", "force-stop", PKG)
    time.sleep(2)
    auto._sh("shell", "monkey", "-p", PKG, "-c",
             "android.intent.category.LAUNCHER", "1")
    time.sleep(8)


def solvable_here():
    """(g, board) if the on-screen puzzle reads & solves, else None."""
    img = auto.screenshot("/tmp/run.png")
    if not play.in_puzzle(img):
        return None
    try:
        g = auto.detect_grid(img, n=None)
        _, _, board = clue_ocr.solvable_clues(img, g)
        return g
    except (AssertionError, RuntimeError):
        return None


def leave_puzzle():
    """Exit the current puzzle to the map (save & exit), non-destructively."""
    auto.back()
    time.sleep(1.2)
    img = auto.screenshot("/tmp/run.png")
    if auto.exit_dialog(img):
        auto.tap(*auto.NAV["confirm_yes"])       # 是 = save & exit
        time.sleep(1.2)


def goto_fresh_puzzle(tries=25):
    """Land in a solvable puzzle: tap unplayed tiles, skipping corrupted ones
    (page forward past them). Returns True if in a solvable puzzle."""
    skipped = 0
    for _ in range(tries):
        img = auto.screenshot("/tmp/run.png")
        if play.in_puzzle(img):
            if solvable_here():
                return True
            leave_puzzle()                        # corrupted -> exit & skip
            auto.swipe(1100, 2545, 300, 2545, ms=400)
            time.sleep(1.0)
            skipped += 1
            continue
        if auto.level_select(img):
            tile = auto.find_next_tile(img)
            if tile:
                auto.tap(*tile)
                time.sleep(1.8)
            else:
                auto.swipe(1100, 2545, 300, 2545, ms=400)   # page forward
                time.sleep(1.0)
            continue
        auto.back()                               # victory/galaxy interstitial
        time.sleep(1.0)
    log(f"goto_fresh_puzzle gave up (skipped {skipped})")
    return False


def main(target):
    done = 0
    since_restart = 0
    fails = 0
    if not goto_fresh_puzzle():
        restart_app()
        goto_fresh_puzzle()
    while done < target:
        if play.play_one():                       # solve current + advance
            done += 1
            since_restart += 1
            fails = 0
            log(f"level complete ({done}/{target})")
            if CHUNK and since_restart >= CHUNK:
                restart_app()
                since_restart = 0
                if not goto_fresh_puzzle():
                    log("could not re-enter after restart")
                    break
        else:
            fails += 1
            log(f"level failed (streak {fails}) -> restart + re-enter")
            restart_app()
            since_restart = 0
            if not goto_fresh_puzzle() or fails >= 4:
                log("repeated failure; stopping")
                break
    log(f"=== DONE: completed {done}/{target} ===")


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    open(LOG, "w").close()
    log(f"run100 start, target={target}")
    main(target)
