
function processTextStream() {
  numFields = WScript.Arguments.Item(0);

  while (!WScript.StdIn.AtEndOfStream) {
    line = WScript.StdIn.ReadLine();
    fields = line.split("|",numFields)
    WScript.Echo(fields.join("|"));
  }
}

// --------------------------------------------------

processTextStream();

