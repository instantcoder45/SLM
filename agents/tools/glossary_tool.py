"""
Computer Architecture glossary / definition lookup tool.

Contains a comprehensive dictionary of 55+ terms from the
'Computer Architecture: A Quantitative Approach' textbook.
Uses fuzzy matching (difflib) to tolerate typos and partial matches.
"""

import difflib
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Glossary database — 55+ terms organised by topic
# ---------------------------------------------------------------------------
GLOSSARY: dict[str, str] = {
    # ── ISA Types ──────────────────────────────────────────────
    "risc": (
        "Reduced Instruction Set Computer. An ISA design philosophy that uses a small, "
        "highly optimised set of fixed-length instructions. Emphasises load/store "
        "architecture, large register files, and simple addressing modes. Examples: "
        "ARM, MIPS, RISC-V."
    ),
    "cisc": (
        "Complex Instruction Set Computer. An ISA design with a rich set of variable-length "
        "instructions that can perform multi-step operations in a single instruction. "
        "Example: x86."
    ),
    "isa": (
        "Instruction Set Architecture. The abstract interface between hardware and "
        "software, defining instructions, registers, data types, addressing modes, "
        "and memory architecture that a processor supports."
    ),
    "risc-v": (
        "An open-source RISC instruction set architecture designed for modularity and "
        "extensibility. Supports base integer instruction sets (RV32I, RV64I) plus "
        "optional extensions (M, A, F, D, C, V)."
    ),
    "mips": (
        "Microprocessor without Interlocked Pipelined Stages. A classic RISC ISA used "
        "extensively in academic teaching. Features a clean 32-bit fixed-length instruction "
        "format with R-, I-, and J-type instructions."
    ),

    # ── Pipeline Concepts ──────────────────────────────────────
    "pipeline": (
        "A technique where multiple instructions are overlapped in execution. The classic "
        "5-stage RISC pipeline has: IF (Instruction Fetch), ID (Instruction Decode), "
        "EX (Execute), MEM (Memory Access), WB (Write Back)."
    ),
    "pipeline hazard": (
        "A situation that prevents the next instruction from executing in its designated "
        "clock cycle. Three types: structural hazards, data hazards, and control hazards."
    ),
    "data hazard": (
        "Occurs when an instruction depends on the result of a prior instruction that "
        "has not yet completed. Sub-types: RAW (Read After Write), WAR (Write After Read), "
        "WAW (Write After Write)."
    ),
    "structural hazard": (
        "Occurs when two instructions need the same hardware resource in the same clock "
        "cycle (e.g., a single-port memory used for both instruction fetch and data access)."
    ),
    "control hazard": (
        "Occurs due to branch instructions; the pipeline may fetch incorrect instructions "
        "before the branch outcome is known. Mitigated by branch prediction, delayed "
        "branches, or speculative execution."
    ),
    "forwarding": (
        "Also called bypassing. A hardware technique that routes a computed result directly "
        "from the pipeline stage where it is produced to the stage where it is needed, "
        "avoiding stalls from data hazards."
    ),
    "stalling": (
        "Inserting a pipeline bubble (NOP) to delay an instruction until a hazard condition "
        "is resolved. Also called a pipeline interlock."
    ),
    "branch prediction": (
        "Hardware mechanism that guesses the outcome of a branch instruction before it is "
        "resolved. Static predictors always predict taken/not-taken; dynamic predictors "
        "use history tables (BHT, BTB) for higher accuracy."
    ),
    "speculative execution": (
        "Executing instructions beyond a branch before knowing whether the branch is taken, "
        "based on a prediction. If mispredicted, the speculative results are discarded "
        "(squashed)."
    ),
    "superscalar": (
        "A processor that can issue and execute multiple instructions per clock cycle "
        "using multiple functional units. Requires sophisticated hazard detection and "
        "dynamic scheduling."
    ),
    "out-of-order execution": (
        "A technique where instructions are executed as soon as their operands are "
        "available, regardless of program order. Uses reservation stations and a "
        "reorder buffer (ROB) to maintain correctness."
    ),
    "tomasulo's algorithm": (
        "A hardware algorithm for dynamic scheduling that uses reservation stations "
        "to track operand availability, enabling out-of-order execution and register "
        "renaming to eliminate WAR and WAW hazards."
    ),
    "reorder buffer": (
        "ROB. A circular buffer that holds the results of speculatively executed "
        "instructions in program order, enabling precise exceptions and in-order "
        "commitment despite out-of-order execution."
    ),

    # ── Memory Hierarchy ───────────────────────────────────────
    "cache": (
        "A small, fast SRAM memory between the CPU and main memory that stores "
        "recently/frequently accessed data. Exploits temporal and spatial locality "
        "to reduce average memory access time."
    ),
    "cache miss": (
        "Occurs when the requested data is not found in the cache. Types: compulsory "
        "(cold), capacity, and conflict misses (the 3 C's)."
    ),
    "direct-mapped cache": (
        "A cache where each memory block maps to exactly one cache line "
        "(line = block address mod number_of_lines). Simple but prone to conflict misses."
    ),
    "set-associative cache": (
        "A cache divided into sets, where each memory block maps to a set and can be "
        "placed in any line within that set. N-way set-associative means N lines per set. "
        "Balances conflict misses and hardware complexity."
    ),
    "fully associative cache": (
        "A cache where a block can be placed in any cache line. Minimises conflict "
        "misses but requires expensive comparator hardware for tag matching."
    ),
    "write-back": (
        "A cache write policy where writes update only the cache; the modified block "
        "is written to main memory only when it is evicted (dirty bit tracks modifications)."
    ),
    "write-through": (
        "A cache write policy where every write updates both the cache and main memory "
        "simultaneously. Simpler but generates more memory traffic."
    ),
    "lru": (
        "Least Recently Used. A cache replacement policy that evicts the block that "
        "has not been accessed for the longest time. Approximations (pseudo-LRU) "
        "are commonly used in hardware."
    ),
    "tlb": (
        "Translation Lookaside Buffer. A small, fast cache that stores recent "
        "virtual-to-physical page translations, speeding up address translation "
        "for virtual memory."
    ),
    "virtual memory": (
        "A technique that provides each process with an illusion of a large, contiguous "
        "address space by mapping virtual addresses to physical addresses using page "
        "tables. Enables memory protection and demand paging."
    ),
    "page table": (
        "A data structure used by the OS to map virtual page numbers to physical frame "
        "numbers. Multi-level page tables reduce memory overhead for sparse address spaces."
    ),
    "amat": (
        "Average Memory Access Time. Calculated as: AMAT = Hit Time + Miss Rate × "
        "Miss Penalty. A key metric for evaluating memory hierarchy performance."
    ),
    "memory wall": (
        "The growing disparity between CPU speed and memory speed/bandwidth. The "
        "processor-memory performance gap has been a major architectural challenge."
    ),

    # ── Parallelism ────────────────────────────────────────────
    "ilp": (
        "Instruction-Level Parallelism. The potential overlap among instructions that "
        "allows multiple instructions to execute simultaneously. Exploited by pipelining, "
        "superscalar, VLIW, and out-of-order execution."
    ),
    "tlp": (
        "Thread-Level Parallelism. Parallelism achieved by running multiple threads or "
        "processes simultaneously on multi-core or multithreaded processors."
    ),
    "dlp": (
        "Data-Level Parallelism. Parallelism achieved by performing the same operation "
        "on multiple data elements simultaneously (e.g., SIMD, vector processing, GPUs)."
    ),
    "simd": (
        "Single Instruction, Multiple Data. A parallelism model where one instruction "
        "operates on multiple data elements simultaneously. Used in vector units, SSE, "
        "AVX, and GPU shader cores."
    ),
    "mimd": (
        "Multiple Instruction, Multiple Data. A parallelism model where multiple "
        "processors execute different instructions on different data. Includes "
        "multicore, SMP, and clusters."
    ),
    "gpu": (
        "Graphics Processing Unit. A massively parallel processor optimised for "
        "data-parallel workloads. Modern GPUs contain thousands of simple cores "
        "and are used for graphics, ML, and scientific computing."
    ),
    "amdahl's law": (
        "Defines the theoretical speedup of a program when only a fraction of it is "
        "parallelised: Speedup = 1 / ((1 - f) + f/N), where f is the parallelisable "
        "fraction and N is the number of processors."
    ),
    "flynn's taxonomy": (
        "A classification of computer architectures based on instruction and data "
        "streams: SISD, SIMD, MISD, MIMD."
    ),
    "multithreading": (
        "A technique where a single processor core can switch between multiple threads "
        "to hide latency. Types: coarse-grained, fine-grained, and simultaneous "
        "multithreading (SMT / Hyper-Threading)."
    ),
    "smt": (
        "Simultaneous Multithreading. A technique allowing a single superscalar core "
        "to issue instructions from multiple threads in the same clock cycle, improving "
        "utilisation of functional units. Intel brands this as Hyper-Threading."
    ),

    # ── Storage & I/O ──────────────────────────────────────────
    "raid": (
        "Redundant Array of Independent Disks. Uses multiple disks to improve "
        "performance and/or reliability. Key levels: RAID 0 (striping), "
        "RAID 1 (mirroring), RAID 5 (striping with parity), RAID 6 (double parity)."
    ),
    "ssd": (
        "Solid-State Drive. A storage device that uses NAND flash memory. Offers "
        "much lower latency and higher throughput than HDDs, with no moving parts."
    ),
    "dma": (
        "Direct Memory Access. A hardware mechanism that allows I/O devices to "
        "transfer data directly to/from memory without CPU intervention, freeing "
        "the CPU for other work."
    ),
    "bus": (
        "A shared communication pathway connecting CPU, memory, and I/O devices. "
        "Characterised by bus width, clock rate, and protocol (synchronous vs "
        "asynchronous)."
    ),
    "interconnect": (
        "The network connecting processors, memories, and I/O in a multiprocessor "
        "system. Topologies include bus, crossbar, ring, mesh, and torus."
    ),

    # ── Performance Metrics ────────────────────────────────────
    "cpi": (
        "Cycles Per Instruction. The average number of clock cycles each instruction "
        "takes to execute. CPU Time = IC × CPI × Clock Cycle Time."
    ),
    "ipc": (
        "Instructions Per Cycle. The reciprocal of CPI; measures how many instructions "
        "a processor completes per clock cycle. Higher IPC indicates better throughput."
    ),
    "throughput": (
        "The number of tasks or instructions completed per unit time. Pipelining "
        "improves throughput (but not individual instruction latency)."
    ),
    "latency": (
        "The time taken to complete a single operation or task from start to finish. "
        "Measured in clock cycles or nanoseconds."
    ),
    "speedup": (
        "The ratio of execution time of the original system to the execution time "
        "of the improved system. Speedup = T_old / T_new."
    ),

    # ── Advanced Concepts ──────────────────────────────────────
    "vliw": (
        "Very Long Instruction Word. An architecture where the compiler packs multiple "
        "independent operations into a single wide instruction, shifting scheduling "
        "complexity from hardware to the compiler."
    ),
    "coherence": (
        "Cache coherence ensures that multiple caches in a multiprocessor system "
        "maintain a consistent view of shared memory. Protocols include snooping "
        "(e.g., MESI) and directory-based schemes."
    ),
    "mesi protocol": (
        "A cache coherence protocol with four states: Modified, Exclusive, Shared, "
        "Invalid. Used in snooping-based multiprocessor systems to maintain coherence."
    ),
    "register renaming": (
        "A technique that eliminates false (WAR, WAW) data dependencies by mapping "
        "architectural registers to a larger set of physical registers."
    ),
    "vector processor": (
        "A processor with special hardware for operating on one-dimensional arrays "
        "(vectors) of data in a pipelined fashion, amortising instruction fetch/decode "
        "overhead across many data elements."
    ),
    "dependability": (
        "The quality of delivered service such that reliance can be placed on this "
        "service. Encompasses reliability, availability, safety, and security."
    ),
}


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------
@tool
def lookup_definition(term: str) -> str:
    """Look up the definition of a Computer Architecture term or concept.

    Use this tool when the user asks for the definition, meaning, or
    explanation of a specific technical term from computer architecture.
    Handles typos and partial matches using fuzzy matching.

    Args:
        term: The term to look up (case-insensitive).
              Examples: 'pipeline', 'TLB', 'Amdahl's law', 'cache miss'

    Returns:
        The definition of the term, or suggestions if no exact match is found.
    """
    try:
        # Normalise input
        query = term.strip().lower()

        # Direct match
        if query in GLOSSARY:
            return f"**{term}**: {GLOSSARY[query]}"

        # Try matching without apostrophes / special chars
        query_clean = query.replace("'", "").replace("'", "").replace("-", " ")
        for key, definition in GLOSSARY.items():
            key_clean = key.replace("'", "").replace("'", "").replace("-", " ")
            if query_clean == key_clean:
                return f"**{term}**: {definition}"

        # Fuzzy matching
        close_matches = difflib.get_close_matches(
            query, list(GLOSSARY.keys()), n=5, cutoff=0.5
        )

        if close_matches:
            # If the best match is very close, return it directly
            best = close_matches[0]
            ratio = difflib.SequenceMatcher(None, query, best).ratio()
            if ratio >= 0.85:
                return f"**{best}** (closest match for '{term}'): {GLOSSARY[best]}"

            # Otherwise list suggestions
            suggestions = ", ".join(f"'{m}'" for m in close_matches)
            return (
                f"Term '{term}' not found in glossary.\n"
                f"Did you mean one of these? {suggestions}\n\n"
                f"Here is the top suggestion:\n"
                f"**{best}**: {GLOSSARY[best]}"
            )

        # Substring search as last resort
        substring_matches = [k for k in GLOSSARY if query in k or k in query]
        if substring_matches:
            results = []
            for match in substring_matches[:3]:
                results.append(f"**{match}**: {GLOSSARY[match]}")
            return "Possible matches:\n\n" + "\n\n".join(results)

        return (
            f"Term '{term}' not found in the glossary. "
            f"The glossary covers: ISA types, pipeline concepts, memory hierarchy, "
            f"parallelism, storage/I/O, and performance metrics. "
            f"Try rephrasing or using a more standard term."
        )

    except Exception as e:
        return f"Glossary lookup error: {type(e).__name__}: {e}"
