"""Long-run picross autoplay.

The fill is reliable (per-run swipe) and the OCR has a line-recovery fallback,
so a healthy run needs no preventive restarts -- we restart the app only when a
level genuinely fails (which also clears any input lag). Saves persist.

All navigation goes through auto.start_next(): a state machine that walks
victory -> galaxy -> level-select -> next tile, cancels the stray exit-confirm
dialog, and -- crucially -- never presses BACK on the level map (that opens the
app-exit dialog) and never touches the in-game RESET (it wipes progress).

    python3 run100.py [target]      # logs to /tmp/run100.log
"""
import sys
import time

import auto
import play

PKG = "com.gmail.joystudio808.nonogram"
LOG = "/tmp/run100.log"
MAX_FAIL = 4                      # consecutive genuine failures -> stop


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


def navigate(skips):
    """Get a puzzle on screen from wherever we are (victory/galaxy/map/dialog),
    skipping `skips` leading unplayed tiles (levels we gave up on). Returns True
    once a puzzle is up. Safe: never backs out of the map."""
    return auto.start_next(wait=1.5, max_steps=25, skips=skips)


def leave_to_map():
    """From a puzzle we're abandoning, back out to the level map (non-
    destructively -- BACK from a puzzle saves & returns to the map; it only
    opens the exit dialog when pressed ON the map, which we cancel just in
    case). NEVER touches the in-game RESET."""
    auto.back()
    time.sleep(1.5)
    img = auto.screenshot("/tmp/run.png")
    if auto.exit_dialog(img):
        auto.tap(*auto.NAV["dialog_cancel"])
        time.sleep(1.2)


def main(target):
    done = skips = nav_fail = 0
    if not navigate(skips):
        restart_app()
        navigate(skips)
    while done < target:
        img = auto.screenshot("/tmp/run.png")
        if not play.in_puzzle(img):
            # between levels (victory/galaxy/map): walk to the next puzzle.
            if navigate(skips):
                nav_fail = 0
                continue
            nav_fail += 1
            log(f"can't reach a puzzle (streak {nav_fail}) -> restart")
            restart_app()
            if not navigate(skips):
                nav_fail += 1
            if nav_fail >= 6:               # app/adb truly dead -> give up
                log("navigation dead; stopping")
                break
            continue
        nav_fail = 0
        if play.solve_current():            # solve only; navigation is loop-top
            done += 1
            log(f"level complete ({done}/{target}, skipped {skips})")
        else:
            # Unsolvable (OCR beyond recovery). Don't stop -- skip it and press
            # on. It stays unplayed at the front, so bumping `skips` makes the
            # next navigation land on the following fresh level.
            skips += 1
            log(f"level UNSOLVABLE -> skip ({skips} skipped, {done} done)")
            leave_to_map()
    log(f"=== DONE: completed {done}/{target}, skipped {skips} ===")


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    open(LOG, "w").close()
    log(f"run100 start, target={target}")
    main(target)
