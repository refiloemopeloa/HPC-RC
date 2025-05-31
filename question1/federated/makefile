CXX = mpic++
CXXFLAGS = -g -Wall -I../helpers
SRCS = main.cpp server.cpp worker.cpp ../helpers/common.cpp
TARGET = federated

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(TARGET)

