def interpret_response(raw_response: str) -> dict:
    """
    Takes raw response from the application and returns a structured result.

    Args:
        raw_response (str): The raw string response, e.g., "crash" or "no crash".

    Returns:
        dict: A structured dictionary indicating crash status.
    """
    response = raw_response.strip().lower()

    if response == "crash":
        return {"status": "ok", "crash": True}
    elif response == "no crash":
        return {"status": "ok", "crash": False}
    else:
        return {
            "status": "error",
            "error": f"Unexpected response: '{raw_response}'"
        }

# Simulating received response from the application
received = "no crash"  # or "no crash", or something unexpected

result = interpret_response(received)

if result["status"] == "ok":
    if result["crash"]:
        print("Application crashed.")
    else:
        print("Application did not crash.")
else:
    print("Error:", result["error"])
