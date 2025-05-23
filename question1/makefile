# Makefile for Federated K-means Learning Framework

CXX = mpicxx
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
INCLUDES = -I./helpers -I./preprocess
LIBS = 

# Source directories
SRC_DIR = .
HELPER_DIR = helpers
PREPROCESS_DIR = preprocess

# Object files
OBJS = main.o server.o worker.o preprocess.o mnist_loader.o image_utils.o common.o

# Target executable
TARGET = federated_kmeans

# Default target
all: $(TARGET)

# Link all object files
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Compile main.cpp
main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile server.cpp
server.o: server.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile worker.cpp
worker.o: worker.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile preprocess.cpp
preprocess.o: $(PREPROCESS_DIR)/preprocess.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile helper files (you'll need to create these)
mnist_loader.o: $(HELPER_DIR)/mnist_loader.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

image_utils.o: $(HELPER_DIR)/image_utils.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

common.o: $(HELPER_DIR)/common.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Baseline K-means (standalone)
baseline: k_baseline.cpp $(HELPER_DIR)/mnist_loader.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o k_baseline k_baseline.cpp $(HELPER_DIR)/mnist_loader.o

# Clean build files
clean:
	rm -f $(OBJS) $(TARGET) k_baseline

# Phony targets
.PHONY: all clean baseline

# Build and run example
run: $(TARGET)
	mpirun -np 4 ./$(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all      - Build the federated learning framework"
	@echo "  baseline - Build the baseline K-means comparison"
	@echo "  clean    - Remove build files"
	@echo "  run      - Build and run with 4 processes (3 workers + 1 server)"
	@echo "  help     - Show this help message"