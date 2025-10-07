import os
import socket
from fuzzer_interface_direct import run_fuzz_cycle_bytes


def SimilarityScore(num_samples1, num_samples2):
    # Handle edge case if both are zero
    if num_samples1 == 0 and num_samples2 == 0:
        return 100.0  # identical
    
    max_val = max(num_samples1, num_samples2)
    diff = abs(num_samples1 - num_samples2)
    
    similarity = (1 - (diff / max_val)) * 100
    return round(similarity, 2)


class Messenger:
    SocketSender = socket.socket()
    restore = []

    def __init__(self,restoreSeed) -> None:
        self.SocketSender = None
        self.restore = restoreSeed

    def DryRunSend(self, squence):
        for message in squence.M:
            response = self.sendMessage(message)
            if response == "#error":
                return None
            squence.R.append(response)

        for message in self.restore.M:
            response = self.sendMessage(message)
            if response == "#error":
                return None

        return squence

    def ProbeSend(self,squence,index): # send a sequence of messages (work for Probe)
        for i in range(len(squence.M)):
            response = self.sendMessage(squence.M[i])
            if response == "#error":
                return "#error"
            elif response == '#crash':
                return '#crash'
            if i == index:
                res = response
        for i in range(len(self.restore.M)):
            resotreResponse = self.sendMessage(self.restore.M[i])
            if resotreResponse == "#error":
                return "#error"
            elif response == '#crash':
                return '#crash'
        return res

    def SnippetMutationSend(self,squence,index): # send a sequence of messages (work for SnippetMutate)
        for i in range(len(squence.M)):
            response = self.sendMessage(squence.M[i])
            if response == "#error":
                return "#error"
            elif response == '#crash':
                return '#crash'
            if i == index:
                res = response

        for i in range(len(self.restore.M)):
            resotreResponse = self.sendMessage(self.restore.M[i])
            if resotreResponse == "#error":
                return "#error"
            elif response == '#crash':
                return '#crash'

        pool = squence.PR[index]
        scores = squence.PS[index]
        #print("+++++")
        #print(res.strip())
        for i in range(len(pool)):
            c = SimilarityScore(pool[i], res)
            #print(pool[i].strip())
            #print(str(c)+"   "+str(scores[i]))
            if c >= scores[i]:
                return ""
        return "#interesting-"+str(index)


    def sendMessage(self, message, retry_time=0):  # send a message (now supports bytes)
        """
        Accepts Message objects where message.raw["Content"] is either:
        - bytes -> send directly
        - str   -> encoded to bytes via latin-1 (1:1 0-255)
        """
        try:
            # 1) Try to get binary content from message.raw["Content"]
            content = None
            if isinstance(message.raw, dict) and "Content" in message.raw:
                content = message.raw["Content"]

                # if it's a str, convert to bytes preserving 0-255 values
                if isinstance(content, str):
                    data_bytes = content.encode("latin-1")
                elif isinstance(content, (bytes, bytearray)):
                    data_bytes = bytes(content)
                else:
                    print("[!] message.raw['Content'] has unsupported type:", type(content))
                    return "#error"

                # call fuzzer bytes API (you need run_fuzz_cycle_bytes exported from fuzzer_interface)
                # I assume an API like: run_fuzz_cycle_bytes(data_bytes) -> (crash_flag, num_samples, chunk_ids)
                try:
                    # prefer bytes-oriented API
                    crash_flag, num_samples, chunk_ids = run_fuzz_cycle_bytes(data_bytes)
                except NameError:
                    # fallback if only run_fuzz_cycle(path) exists in your module:
                    # write a small temp file and call existing function (less ideal).
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False) as tf:
                        tf.write(data_bytes)
                        tf.flush()
                        tfname = tf.name
                    try:
                        crash_flag, num_samples, chunk_ids = run_fuzz_cycle_bytes(tfname)
                    finally:
                        try:
                            os.unlink(tfname)
                        except Exception:
                            pass

                if crash_flag == 1:
                    return "#crash"
                else:
                    # keep behavior: return numeric/summary (your code expects number sometimes)
                    return num_samples

        except Exception as e:
            print(f"[!] Exception during fuzz cycle: {e}")
            return "#error"



        
