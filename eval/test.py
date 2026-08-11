from main.retrieval.retrieval import search

failing_questions = {
    3: ("How many dedicated IO pins does the RP2040 have for SPI Flash?", [15]),
    4: ("What are the ADC specs on the RP2040?", [2211, 2213]),
    5: ("What USB standard does the RP2040 support?", [15]),
    11: ("What does XIN also function as if XOUT is disconnected?", [18]),
    13: ("Does the RP2040 have onboard non-volatile flash storage?", [14]),
}

for idx, (question, expected) in failing_questions.items():
    retrieved = search(
        "hybrid_reranked", question, k=5
    )  # or whatever your method arg is
    hit = bool(set(retrieved) & set(expected))
    print(f"Q{idx}: expected {expected}, got {retrieved}, HIT: {hit}")
