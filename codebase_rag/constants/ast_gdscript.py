# GDScript (Godot) tree-sitter node types.
# Grammar: PrestonKnopp/tree-sitter-gdscript (Godot 4.x).
# File root is `source`. Nested classes are `class_definition`; the
# file-level `class_name Foo` statement is `class_name_statement`.
# Constructors are always `_init` (`constructor_definition` has no name
# field). Calls are `call` (free/callee expression), `attribute_call`
# (`obj.method(...)` inside an `attribute` chain), and `base_call`
# (`super.method(...)`).

TS_GDSCRIPT_SOURCE = "source"
TS_GDSCRIPT_FUNCTION_DEFINITION = "function_definition"
TS_GDSCRIPT_CONSTRUCTOR_DEFINITION = "constructor_definition"
TS_GDSCRIPT_LAMBDA = "lambda"
TS_GDSCRIPT_CLASS_DEFINITION = "class_definition"
TS_GDSCRIPT_CLASS_NAME_STATEMENT = "class_name_statement"
TS_GDSCRIPT_ENUM_DEFINITION = "enum_definition"
TS_GDSCRIPT_CALL = "call"
TS_GDSCRIPT_ATTRIBUTE_CALL = "attribute_call"
TS_GDSCRIPT_BASE_CALL = "base_call"
TS_GDSCRIPT_ATTRIBUTE = "attribute"
TS_GDSCRIPT_EXTENDS_STATEMENT = "extends_statement"
TS_GDSCRIPT_NAME = "name"
TS_GDSCRIPT_IDENTIFIER = "identifier"
TS_GDSCRIPT_ARGUMENTS = "arguments"
TS_GDSCRIPT_SIGNAL_STATEMENT = "signal_statement"

# Godot constructor name is always `_init`; the grammar node has no `name`.
GDSCRIPT_INIT_NAME = "_init"
