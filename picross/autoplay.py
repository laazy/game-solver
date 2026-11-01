"""Autonomous, self-healing picross autoplay (background-friendly).

Recovery rules, all learned from real failures:

  * Device input lag. The game queues touches and, after long hammering, lags --
    fills drop cells or coalesce into X marks. Cure: RESTART the app (saves
    persist), which clears the lag. Triggered only when a board can't be read
    at all, or a fill leaves a huge fraction of cells unfilled.
  * Never press BACK on the level map. Back at the map root opens the
    "exit the app?" dialog; pressing on blindly there cascaded us out. So we
    detect the map robustly and only ever TAP tiles there.
  * Always cancel any dialog (取消) before deciding anything else.
  * A corrupted in-progress board (X-marked from an earlier bad run) can't be
    read even after a restart -> save&exit and page the tile strip forward so
    we move past it instead of re-entering it.

Progress -> /tmp/autoplay.log.   Run:  python3 autoplay.py [target]
"""
import sys
import time

from PIL import Image

import auto
import play

PKG = "com.gmail.joystudio808.nonogram"
LOG = "/tmp/autoplay.log"


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def shot():
    return auto.screenshot("/tmp/auto.png")


def restart_app():
    log("restart app (clear input lag)")
    auto._sh("shell", "am", "force-stop", PKG)
    time.sleep(2)
    auto._sh("shell", "monkey", "-p", PKG, "-c",
             "android.intent.category.LAUNCHER", "1")
    time.sleep(8)


def is_map(img):
    """Robust: the bottom tile strip shows several tiles (played or not), each
    with light content over the dark sky. >=4 -> we're on the level map."""
    px = Image.open(img).convert("RGB").load()
    tiles = sum(1 for y in auto.NAV["tile_ys"] for x in auto.NAV["tile_xs"]
                if auto._warm_frac(px, x, y) > 0.03)
    return tiles >= 4


def next_tile(img):
    """First unplayed (cool) visible tile, else None."""
    px = Image.open(img).convert("RGB").load()
    for y in auto.NAV["tile_ys"]:
        for x in auto.NAV["tile_xs"]:
            if auto._warm_frac(px, x, y) < auto.NAV["tile_open_frac"]:
                return (x, y)
    return None


def page_forward():
    auto.swipe(1100, 2545, 300, 2545, ms=400)
    time.sleep(1.0)


def cancel_dialog(img):
    if auto.exit_dialog(img):
        auto.tap(*auto.NAV["dialog_cancel"])        # 取消 (stay in app/puzzle)
        time.sleep(1.0)
        return True
    return False


def try_solve():
    """Solve+fill the on-screen puzzle. -> 'won' | 'lag' | 'unreadable'."""
    res = play.read_solvable()
    if not res:
        return "unreadable"
    g, board = res
    n = g["n"]
    nfill = sum(v == 1 for r in board for v in r)
    auto.fill(board, g, fill_button=None, swipe_runs=True)
    if play._won():
        return "won"
    actual = auto.read_board("/tmp/play.png", g, n=n)
    drops = [(i, j) for i in range(n) for j in range(n)
             if board[i][j] == 1 and actual[i][j] != 1]
    if len(drops) > nfill * 0.4:                    # gross failure -> lag
        return "lag"
    for i, j in drops:                              # small drops -> repair
        auto.tap(*auto.cell_center(g, i, j))
        time.sleep(0.08)
    return "won" if play._won() else "lag"


def exit_to_map():
    """Leave a puzzle (save & exit), then page forward past corrupted tiles."""
    auto.back()
    time.sleep(1.0)
    img = shot()
    if auto.exit_dialog(img):
        auto.tap(*auto.NAV["confirm_yes"])          # 是 = save & exit
        time.sleep(1.2)


def main(target):
    done = restarts = stale = bad_boards = 0
    while done < target and restarts < 10:
        img = shot()

        if cancel_dialog(img):
            continue

        if play.in_puzzle(img):
            r = try_solve()
            if r == "won":
                done += 1
                stale = 0
                log(f"level complete ({done}/{target})")
                auto.start_next()
            elif r == "lag":
                restart_app()
                restarts += 1
            else:                                   # unreadable -> corrupted
                bad_boards += 1
                log(f"unreadable board #{bad_boards}; save&exit + page forward")
                exit_to_map()
                page_forward()
            continue

        if is_map(img):
            stale = 0
            tile = next_tile(img)
            if tile:
                auto.tap(*tile)
                time.sleep(1.8)
            else:
                page_forward()                       # no fresh tile here
            continue

        # victory / galaxy interstitial: BACK is safe here (we're above the map
        # root, not on it). Nudge toward the map, then a fresh tile.
        auto.back()
        time.sleep(1.0)
        stale += 1
        if stale > 6:                                # wedged -> restart
            restart_app()
            restarts += 1
            stale = 0

    log(f"=== DONE: completed {done}/{target} (restarts={restarts}) ===")


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    open(LOG, "w").close()
    log(f"autoplay start, target={target}")
    main(target)
