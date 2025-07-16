#  Local RAG with LLaMA.cpp

This project is a modified fork of the original repository which is aimed at **local Retrieval-Augmented Generation (RAG)**,by designed to demonstrate **local RAG** using a **quantized GGUF model** (specifically, LLaMA 2 7B Instruct) — entirely offline and locally on your system.

---

##  Data Used

The source document is a PDF file containing **lecture notes on the 8085 Microprocessor (MPU)**.

---

##  Setup & Usage

### 0. Install dependencies using:

```bash
pip install -r requirements.txt
```

---

### 1. Populate the Vector Database

Convert the raw PDF into chunked embeddings:

```bash
python3 populate_database.py
```

This will process the PDF and store vector embeddings for efficient retrieval.

---

### 2. Query the Model

Ask questions related to the content of the PDF:

```bash
python3 query_data.py "your question here"
```

####  Example output:

```bash
python3 query_data.py "Summarize what a flag register does"
```

####  Sample Output:

```json
{
  "Response": {
    "text": "The flag register in the 8085 microprocessor is a group of five flip-flops (Z, AC, P, S, and C) that are used to indicate the result of an arithmetic or logical operation. These flags are set or reset based on the result of the operation and are used to control the flow of instructions. The flag register is used to provide data condition information about the accumulator, and it is an essential component of the ALU (Arithmetic Logic Unit)."
  },
  "Sources": [
    "data/8085 MPU full.pdf:22:0",
    "data/8085 MPU full.pdf:23:0",
    "data/8085 MPU full.pdf:24:0",
    "data/8085 MPU full.pdf:68:0",
    "data/8085 MPU full.pdf:70:0"
  ]
}
```

---


## Notes

- Everything runs **locally**, no internet or cloud APIs required.
- Designed for educational and experimentation purposes with RAG + LLM.


---
