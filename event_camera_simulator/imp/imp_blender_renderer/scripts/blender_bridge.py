import argparse
import logging
import zmq
import bpy
import mathutils


TEMP_VARIABLE_NAME = "temp_"
LOGGING_FORMAT_STR = "[%(asctime)s] %(message)s"
INITIALIZED_LOG_FORMAT_STR = "Server initialized at TCP port {}"
ENDPOINT_FORMAT_STR = "tcp://127.0.0.1:{:d}"
CMD_LOG_FORMAT_STR = "COMMAND: {}"
RETURN_LOG_FORMAT_STR = "RETURN: {}"
ERROR_LOG_FORMAT = "ERROR:"
SUCCESS_MSG = b"OK"
ERROR_MSG = b"ERROR"


def exec_with_return(command, globals, locals):
    command_with_return = "{} = {}".format(TEMP_VARIABLE_NAME, command) 
    exec(command_with_return, globals, locals)
    return locals[TEMP_VARIABLE_NAME]

def main(args):
    logging.basicConfig(format=LOGGING_FORMAT_STR)
    logging.getLogger().setLevel(logging.INFO)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(ENDPOINT_FORMAT_STR.format(args.port))
    logging.info(INITIALIZED_LOG_FORMAT_STR.format(args.port))

    while True:
        command = socket.recv().decode("utf-8")
        logging.info(CMD_LOG_FORMAT_STR.format(command))
        try:
            returned = exec_with_return(command, globals(), locals())
            logging.info(RETURN_LOG_FORMAT_STR.format(returned))
            socket.send_multipart([ SUCCESS_MSG, str(returned).encode("utf-8") ])
        except KeyboardInterrupt:
            socket.close()
            context.term()
            return
        except Exception as e:
            logging.exception(ERROR_LOG_FORMAT)
            socket.send_multipart([ ERROR_MSG, str(e).encode("utf-8") ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Server script for executing bpy commands.")
    )
    parser.add_argument(
        "--port", type=int, default=5555, help="TCP port number of the server."
    )
    args = parser.parse_args()

    main(args)
