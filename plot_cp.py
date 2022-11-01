import argparse
import os
import sys
from glob import glob
import logging
import subprocess
from tqdm import tqdm


def get_plotter():
    plotter_paths = glob('/dev/usb/lp*')
    assert len(plotter_paths) > 0, f'Found no plotters at "/dev/usb/"'
    assert len(plotter_paths) <= 1, f'Found multiple plotters: {plotter_paths}'
    return plotter_paths[0]


def chunked(size, source):
    for i in range(0, len(source), size):
        yield source[i:i+size]


def plot_file(path, plotter=None):
    plotter = get_plotter() if plotter is None else plotter
    with open(path, 'rb') as f:
        data = f.read()
    with open(plotter, 'wb') as lp:
        [lp.write(bytes(b)) for b in tqdm(list(chunked(128, data)), desc=f'Plotting {path}..')] #tqdm(f.read())]


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('plotting')
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument('dirs', type=str, nargs='+',
                        help='Directories containing the CPs to plot. '
                             'Expects a combination of "cp_for_cutting.svg", "cp_for_cutting_interior.svg", '
                             '"cp_for_cutting_borders.svg" or the corresponding .hpgl files.')
    parser.add_argument('-s', '--score', action='store_true',
                        help='If true, scores the crease pattern and its border.')
    parser.add_argument('-d', '--draw_creases', action='store_true',
                        help='If true, draws the (interior) creases.')
    parser.add_argument('-b', '--draw_borders', action='store_true',
                        help='If true, draws the borders.')
    parser.add_argument('-c', '--convert', action='store_true',
                        help='If true, automatically converts .svg to .hpgl if needed')
    parser.add_argument('-p', '--plotter', type=str, default=None,
                        help='Path to plotter. By default, will look at /dev/usb/lp*.')
    parser.add_argument('-r', '--repetitions', type=int, default=1,
                        help='Number of repetitions, each file will be plotted this many times.')

    args = parser.parse_args()

    if args.score or args.draw_creases or args.draw_borders:
        # find plotter
        if args.plotter is None:
            args.plotter = get_plotter()

        # logger.info(f"Found plotter at {args.plotter}")
        print(f"Found plotter at {args.plotter}")

    # subprocess.run(["cat", "test.hpgl", ">", "test5.hpgl"])
    # lpr =  subprocess.Popen("/usr/bin/lpr", stdin=subprocess.PIPE)
    # lpr.stdin.write(your_data_here)

    print(f'{len(args.dirs)} directories will be processed:')
    for directory in args.dirs:
        assert os.path.isdir(directory), f'{directory}'
        print(directory)
    print()

    for i, directory in enumerate(args.dirs):
        print(f'Processing {directory} ({i+1}/{len(args.dirs)})')
        directory = os.path.expanduser(directory)

        files = [os.path.join(directory, f)
                 for f in ['cp_for_cutting', 'cp_for_cutting_interior', 'cp_for_cutting_borders']]
        svg_files = [f + '.svg' for f in files]
        hpgl_files = [f + '.hpgl' for f in files]

        for file in svg_files + hpgl_files:
            if os.path.exists(file):
                print(f'Found {file}')

        if args.convert:
            for svg_file, hpgl_file in zip(svg_files, hpgl_files):
                if os.path.exists(svg_file) and not os.path.exists(hpgl_file):
                    print(f'Converting {svg_file} to {hpgl_file}.')
                    subprocess.run(["inkscape", svg_file, f"--export-filename={hpgl_file}"])

        for _ in range(args.repetitions):

            if args.score:
                # find the files to plot
                if os.path.exists(hpgl_files[0]):
                    files_to_score = [hpgl_files[0]]
                else:
                    assert os.path.exists(hpgl_files[1]) and os.path.exists(hpgl_files[2]), \
                        f'Some files for scoring are missing: ' \
                        f'{[(f, os.path.exists(f)) for f in hpgl_files]}'
                    files_to_score = [hpgl_files[2], hpgl_files[1]]
                input("\nMake sure that the KNIFE is inserted. Confirm when ready.")
                for f in files_to_score:
                    plot_file(hpgl_files[0], plotter=args.plotter)

            # drawing
            if args.draw_creases and args.draw_borders:
                # find the files to plot
                if os.path.exists(hpgl_files[0]):
                    files_to_draw = [hpgl_files[0]]
                else:
                    assert os.path.exists(hpgl_files[1]) and os.path.exists(hpgl_files[2]), \
                        f'Some files for drawing are missing: ' \
                        f'{[(f, os.path.exists(f)) for f in hpgl_files]}'
                    files_to_draw = [hpgl_files[1:]]
            else:
                files_to_draw = []
                if args.draw_borders:
                    files_to_draw.append(hpgl_files[2])
                if args.draw_creases:
                    files_to_draw.append(hpgl_files[1])
            if files_to_draw:
                input("\nMake sure that the PEN is inserted. Confirm when ready.")
                for f in files_to_draw:
                    plot_file(f, plotter=args.plotter)
