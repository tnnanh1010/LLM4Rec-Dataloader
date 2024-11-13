import itertools
import pandas as pd

def uir_parser(tokens, **kwargs):
    return [(tokens[0], tokens[1], float(tokens[2]))]

def uirt_parser(tokens, **kwargs):
    return [(tokens[0], tokens[1], float(tokens[2]), int(tokens[3]))]

PARSERS = {
    "UIR": uir_parser,
    "UIRT": uirt_parser,
}

class Reader:
    def __init__(self,
                 encoding="utf-8"):
        self.encoding = encoding
    
    def read(self,
             path,
             format="UIRT",
             sep="\t",
             skip_lines=0,
             id_inline=False,
             parser=None):
        parser = PARSERS.get(format) if parser is None else parser
        with open(path, encoding = self.encoding) as f:
            tuples = [
                tup
                for idx, line in enumerate(itertools.islice(f, skip_lines, None))
                for tup in parser(
                    line.strip().split(sep), line_idx=idx, id_inline=id_inline
                )
            ]
        df = pd.DataFrame(tuples, columns=["user_id", "item_id", "rating", "timestamp"]) 
        return df
        # return tuples

if __name__ == "__main__":
    reader = Reader()
    data = reader.read("ml-1m/ratings.dat", format="UIRT", sep="::")
    print(data[:5])