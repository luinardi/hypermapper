CXX=g++
CFLAGS=-std=c++17 -g -Wall
LDFLGS=

LIBS=-lstdc++fs

OBJ = cpp_client.o


%.o: %.cpp
	$(CXX) -c -o $@ $< $(CFLAGS)

cpp_client: $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm *.o cpp_client
