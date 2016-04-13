CC       = pgcc
CCFLAGS  = 
ACCFLAGS = -acc -ta=nvidia -Minfo=accel
OMPFLAGS = -fast -mp -Minfo

BIN =  acc omp

all: $(BIN)

acc: mm_acc.c
	$(CC) $(CCFLAGS) $(ACCFLAGS) -o $@ $<

omp: mm_acc.c
	$(CC) $(CCFLAGS) $(OMPFLAGS) -o $@ $<

clean:
	$(RM) $(BIN)
