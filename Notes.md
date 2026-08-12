### Problems encountered
- Lexical Gap: 
  - The BM25 and vector search pulled out chunks with the most amount of words, without considered context. i.e. when asked about "What USB standard does the RP2040 support?", it pulled chunks from the docs about USBCTRL and USB device boot, ranking the chunk with the word USB appearing once lower (even if it was the right one)













### Tasks:
1. IMPROVE GEN ACCURACY
  - Fix pipline failiures (ensuring there are virtually 0 IDK's if the context exists, which was a problem I ran into while running the eval)
  - Test long context questions
2. Wrap it into a CLI App (maybe get claude to make a nice TUI 😋)
3. Improve performance (future considerations)
