CXX = mpic++
CXXFLAGS = -g -Wall -I../helpers
SRCS = main.cpp server.cpp worker.cpp ../helpers/common.cpp
TARGET = federated

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(TARGET)
	@if [ -z "$(NP)" ]; then \
		echo "Usage: make run NP=3"; \
	else \
		mpirun --oversubscribe -np $(NP) ./$(TARGET); \
	fi

clean:
	rm -f $(TARGET)

