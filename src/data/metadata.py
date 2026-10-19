import src.helpers.verbose as Verbose

_TAG = "METADATA"

METADATA_FILE = "data\\METADATA.DATA"
metadata_file_buffer = None
metadata_content_buffer = None


def Configure():

    global metadata_file_buffer
    global metadata_content_buffer

    if metadata_file_buffer != None:
        Verbose.Write(
            _TAG,
            "Error: Configuration buffer already open...",
            Verbose.CONSOLE_STATUS.ERROR,
        )
        return False

    # CREATE IF NOT EXIST
    metadata_file_buffer = open(METADATA_FILE, "a+", encoding="utf-8")
    metadata_file_buffer.seek(0)
    metadata_content_buffer = metadata_file_buffer.read().splitlines()
    Verbose.Write(_TAG, "Configuration completed.", Verbose.CONSOLE_STATUS.SUCCESS)


def Write(content):
    metadata_file_buffer.write(content + "\n")
    metadata_file_buffer.flush()
    metadata_content_buffer.append(content)


def Read():

    if not metadata_content_buffer:
        return []

    return list(metadata_content_buffer)


def Rewrite(lines):

    global metadata_content_buffer

    if metadata_file_buffer is None or metadata_file_buffer.closed:
        return False

    metadata_file_buffer.seek(0)
    metadata_file_buffer.truncate()
    metadata_file_buffer.write("".join(line + "\n" for line in lines))
    metadata_file_buffer.flush()
    metadata_content_buffer = list(lines)

    return True


def Find(content):
    if metadata_content_buffer == None:
        Verbose.Write(
            _TAG,
            "Error: Cannot access metadata file content...",
            Verbose.CONSOLE_STATUS.ERROR,
        )
        return False

    return content in metadata_content_buffer


def EraseAll():

    if metadata_file_buffer is None:
        return False

    if metadata_file_buffer is not None and not metadata_file_buffer.closed:
        metadata_file_buffer.seek(0)
        metadata_file_buffer.truncate()
        metadata_file_buffer.flush()
        metadata_content_buffer.clear()

    return True


def Close():

    global metadata_content_buffer

    if metadata_file_buffer == None:
        Verbose.Write(
            _TAG,
            "Error: Cannot access metadata file content...",
            Verbose.CONSOLE_STATUS.ERROR,
        )
        return False

    metadata_file_buffer.close()
    metadata_content_buffer = ""