CXX=g++
CFLAGS=-std=c++17 -g 
LDFLGS=

LIBS=-lstdc++fs

OBJ = c_client.o


%.o: %.cpp
	$(CXX) -c -o $@ $< $(CFLAGS)

c_client: $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm *.o c_client 
