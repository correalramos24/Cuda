NVCC        = nvcc
NVCC_FLAGS  = -g
OBJ         = file.o main.o kernel.o
EXE         = vecAdd

default: $(EXE)

file.o: ../common/file.c ../common/file.h
	$(NVCC) $(NVCC_FLAGS) -c ../common/file.c -o $@

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)