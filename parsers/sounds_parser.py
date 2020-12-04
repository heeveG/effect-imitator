import os


def get_params(filename):
    params = filename.split("-")
    return {"guitar type": params[0],
            "note": params[1][:2],
            "picking_type": params[1][2:], 
            "ef_intensity": params[2][3]
            }


def parse_sound(verifier, effect_dir):
    return [filename for filename in os.listdir(effect_dir) 
            if verifier(get_params(filename))]
        