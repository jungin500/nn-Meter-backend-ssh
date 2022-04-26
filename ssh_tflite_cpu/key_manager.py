from paramiko import SSHClient, AutoAddPolicy, RSAKey, ECDSAKey, Ed25519Key, SSHException

def parse_keyfile(filename, password=None):
    pkey = None
    
    try:
        pkey = RSAKey.from_private_key_file(
            filename=filename, password=password
        )
    except SSHException:
        try:
            pkey = ECDSAKey.from_private_key_file(
                filename=filename, password=password
            )
        except SSHException:
            try:
                pkey = Ed25519Key.from_private_key_file(
                    filename=filename, password=password
                )
            except SSHException:
                pass
            
    return pkey