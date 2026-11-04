from __future__ import annotations

from tree_sitter import Node

from ... import constants as cs


def gdscript_get_name(node: Node) -> str | None:
    # constructor_definition is always Godot `_init` and has no `name` field.
    if node.type == cs.TS_GDSCRIPT_CONSTRUCTOR_DEFINITION:
        return cs.GDSCRIPT_INIT_NAME
    if node.type == cs.TS_GDSCRIPT_SOURCE:
        return None
    name_node = node.child_by_field_name(cs.FIELD_NAME)
    if name_node and name_node.text:
        return name_node.text.decode(cs.ENCODING_UTF8)
    for field_name in cs.NAME_FIELDS:
        name_node = node.child_by_field_name(field_name)
        if name_node and name_node.text:
            return name_node.text.decode(cs.ENCODING_UTF8)
    return None


def _node_text(node: Node) -> str | None:
    if node.text is None:
        return None
    return node.text.decode(cs.ENCODING_UTF8)


def _last_identifier(node: Node) -> str | None:
    found: list[str] = []

    def walk(n: Node) -> None:
        if n.type == cs.TS_GDSCRIPT_IDENTIFIER and n.text:
            found.append(n.text.decode(cs.ENCODING_UTF8))
        for child in n.named_children:
            walk(child)

    walk(node)
    return found[-1] if found else _node_text(node)


def gdscript_call_name(call_node: Node) -> str | None:
    """Callee simple/dotted name for a GDScript call site.

    `foo()` is a `call` whose first non-arguments child is the callee.
    `obj.method()` is an `attribute_call` whose identifier child is `method`.
    `super.method()` is a `base_call` with the same identifier child.
    """
    if call_node.type in (
        cs.TS_GDSCRIPT_ATTRIBUTE_CALL,
        cs.TS_GDSCRIPT_BASE_CALL,
    ):
        for child in call_node.named_children:
            if child.type == cs.TS_GDSCRIPT_IDENTIFIER and child.text:
                return child.text.decode(cs.ENCODING_UTF8)
        return None

    if call_node.type != cs.TS_GDSCRIPT_CALL:
        return None

    for child in call_node.named_children:
        if child.type == cs.TS_GDSCRIPT_ARGUMENTS:
            continue
        if child.type == cs.TS_GDSCRIPT_IDENTIFIER:
            return _node_text(child)
        if child.type == cs.TS_GDSCRIPT_ATTRIBUTE:
            return _last_identifier(child)
        text = _node_text(child)
        if text:
            return text.split(".")[-1] if "." in text else text
        return _last_identifier(child)
    return None
