import pickle
from collections import OrderedDict


def main():
	d = OrderedDict()
	pickle.dump(d,open("mapping.p","wb"))

if __name__ == '__main__':
	main()
