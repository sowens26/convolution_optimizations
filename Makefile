all: 
	g++ sequential_basic.cpp -o sequential_basic
	g++ sequential_opt.cpp -o sequential_opt
	g++ multi_basic.cpp -o multi_basic
	g++ multi_opt.cpp -o multi_opt
clean:
	rm -f sequential basic sequential_opt multi_basic multi_opt
sequential_basic:
	g++ sequential_basic.cpp -o sequential_basic
sequential_opt:
	g++ sequential_opt.cpp -o sequential_opt
multi_basic:
	g++ multi_basic.cpp -o multi_basic
multi_opt:
	g++ multi_opt.cpp -o multi_opt