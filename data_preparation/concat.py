import wave
from parsers import sounds_parser
import argparse


def main(args):
    datain = []
    data = []
    out_verifier = lambda p: True if p["ef_intensity"] == "2" \
                                     and p["guitar type"] == "G61" \
                                     and int(p["note"]) in range(40, 66) \
        else False

    infiles = sounds_parser.parse_sound(out_verifier, args.prefix_in)
    print(infiles)
    for infile in infiles:
        nofx = sounds_parser.parse_sound(lambda x: x["guitar type"] == "G61" \
                                                   and x["note"] == sounds_parser.get_params(infile)["note"] \
                                                   and x["picking_type"] == sounds_parser.get_params(infile)[
                                                       "picking_type"], args.nofx_prefix)[0]
        w = wave.open(args.nofx_prefix + nofx, 'rb')
        z = wave.open(args.prefix_in + infile, 'rb')
        datain.append([w.getparams(), w.readframes(w.getnframes())])
        data.append([z.getparams(), z.readframes(z.getnframes())])
        z.close()
        w.close()

    output_in = wave.open(args.infile, 'wb')
    output = wave.open(args.outfile, 'wb')
    output_in.setparams(datain[0][0])
    output.setparams(data[0][0])
    for i in range(len(data)):
        output.writeframes(data[i][1])
    for i in range(len(datain)):
        output_in.writeframes(datain[i][1])
    output.close()
    output_in.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nofx_prefix", default="../../data/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon/Samples/NoFX/")
    parser.add_argument("--prefix_in", default="../../data/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon/Samples/Flanger/")
    parser.add_argument("--infile", default="../data/training/in/sounds_in.wav")
    parser.add_argument("--outfile", default="../data/training/out/sounds_out.wav")
    args = parser.parse_args()
    main(args)
