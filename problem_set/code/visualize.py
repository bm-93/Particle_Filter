from map_reader import *
import sys

def main():
    map_ = sys.argv[1]
    reader = MapReader(map_)
    reader.visualize_map()

if __name__ == "__main__":
    main()