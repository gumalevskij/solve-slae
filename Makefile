TARGET = SolveMatrix

CXX = g++

#EIGENDIR = /home/gumalevskii/Interviews/Arhitex/eigen

CXXFLAGS = -I $(EIGENDIR) -O3 -std=c++17 -lopenblas -llapacke -llapack

SRC = SolveMatrix.cpp

all: clean $(TARGET)

$(TARGET):
	$(CXX) -o $(TARGET) $(SRC) $(CXXFLAGS)  

clean:
	rm -f $(TARGET)
