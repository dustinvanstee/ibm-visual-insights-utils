import paiv_utils as paiv
import argparse as ap

class SmartFormatterMixin(ap.HelpFormatter):
    # ref:
    # http://stackoverflow.com/questions/3853722/python-argparse-how-to-insert-newline-in-the-help-text
    # @IgnorePep8

    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('S|'):
            return text[2:].splitlines()
        return ap.HelpFormatter._split_lines(self, text, width)


class CustomFormatter(ap.RawDescriptionHelpFormatter, SmartFormatterMixin):
    '''Convenience formatter_class for argparse help print out.'''


def _parser():
    parser = ap.ArgumentParser(description='Tool to annotate a video using PowerAI Vision '
                                           'Requires :'
                                           'Requires :'
                                           'Requires :'
                                           '  python score_exported_dataset.py --validate_mode=classification --model_url=https://129.40.2.225/powerai-vision/api/dlapis/8f80467f-470c-47f3-bf3c-ab7e0880a66b --data_directory=/data/work/osa/2018-10-PSEG/datasets_local/dv_97_classification_augmented_dataset-test',
                               formatter_class=CustomFormatter)

    parser.add_argument(
        '--input_video', action='store', nargs='?', required=True,
        help='S|--input_video=<video file name>'
             'Default: %(default)s')

    parser.add_argument(
        '--model_url', action='store', nargs='?',
        required=True,
        help='S|--model_url=<deployed model endpoint>')

    parser.add_argument(
        '--output_directory', action='store', nargs='?',
        required=True,
        help='S|--data_directory=<location of exported PAIV dataset>')

    parser.add_argument(
        '--output_filename', type=str,
        required=False, default="custom.mov",
        help='S|--data_directory=<location of exported PAIV dataset>')

    parser.add_argument('--force_refresh', dest='force_refresh', action='store_true',
                        help='S|--force_refresh=[True|False] '
                        'Default: %(default)s)')
    parser.set_defaults(force_refresh=False)

    parser.add_argument(
        '--sample_rate', type=int, default=100, required=False,
        help='S|Frame sample rate.  sample_rate = 2 means sample at 2X rate'
             'Default: %(default)s')

    args = parser.parse_args()

    return args


def main():
    # Parse command line argument
    args = _parser()
    # --model_url='https://129.40.2.225/powerai-vision/api/dlapis/bda90858-45e4-4ca6-8161-7d63436bb6c6' --input_video="/data/work/osa/2018-10-PSEG/Videos/transmission\ tower\ detection\ demo.mp4" --output_directory="/data/work/osa/2018-10-PSEG/Videos/temp"

    #def edit_video(input_video, model_url,output_directory, output_fn, overlay_fn, max_frames=50, force_refresh=True):

    #print(args)
    #args.force_refresh = True
    for argk in vars(args) :
        paiv.nprint("{} {}".format(argk,vars(args)[argk]))

    paiv.edit_video_objdet(input_video=args.input_video, model_url=args.model_url, output_directory=args.output_directory\
                           ,output_fn=args.output_filename, force_refresh=args.force_refresh, max_frames=5000, sample_rate=args.sample_rate, counter_mode="screen_time")

    paiv.nprint("Program Finished")
    for argk in vars(args) :
        paiv.nprint("{} {}".format(argk,vars(args)[argk]))

if __name__== "__main__":
  main()

# Todo : add threading for quicker video building
# Todo : add custom logic for tracking ball touches with denoising / smoothing
# Todo : add custom logic for displaying number of players at any given time with denoising / smoothing
