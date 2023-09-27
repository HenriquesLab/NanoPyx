import os
import nbformat


def swap_tags(data: str, tags: str) -> str:
    """
    Replaces tags in a string with new values.

    Args:
        data: A string containing the tags to be replaced.
        tags: A string containing the original and new tags in the format
            "original1,new1;original2,new2;..."

    Returns:
        A string with the replaced tags.
    """
    split_tags = tags.split(";")

    # If there is at least one tag to be replaced
    if len(split_tags[0].split(",")) > 1:
        for tags in split_tags:
            original = tags.split(",")[0]
            new = tags.split(",")[1]
            data = data.replace(original, new)

    return data


def check_tag2tag(cell_line):
    """
    Check if a cell line contains the tag2tag string.
    :param cell_line: str, a line of text from a cell in a table.
    :return: bool, True if the line contains the tag2tag string, False otherwise.
    """
    # Split the line by the tag2tag string
    parts = cell_line.split("$tag2tag: ")

    # If the resulting list has more than one element, the tag2tag string is present
    if len(parts) > 1:
        return True

    # Otherwise, the tag2tag string is not present
    return False


def get_include(tag: str, ingredient_path: str) -> str:
    """
    Reads a file with the given tag from the given path and returns its content as a string.

    Args:
        tag: A string representing the filename to be read.
        ingredient_path: A string representing the directory containing the file.

    Returns:
        A string containing the content of the file.
    """
    with open(os.path.join(ingredient_path, tag)) as f:
        content = f.read()
        if not content.endswith("\n"):
            content += "\n"
        return content


def check_include(cell_line):
    """
    This function checks whether a cell line starts with the '$include: ' string.

    Args:
        cell_line (str): The string to be checked.

    Returns:
        bool: True if the string starts with '$include: ', False otherwise.
    """
    return cell_line.startswith("$include: ")


def parse_recipe(recipe: str, notebook: dict, ingredient_path: str, output_path: str) -> None:
    """
    This function takes a recipe file in the specified format and creates a Jupyter notebook.
    :param recipe: str - The path to the recipe file.
    :param notebook: dict - The dictionary representation of the Jupyter notebook.
    :param ingredient_path: str - The path to the directory containing the ingredient files.
    :param output_path: str - The path where the parsed notebook should be saved.
    """

    # Clear all existing cells in the notebook
    notebook["cells"] = []

    # Get the name of the recipe to use it as the notebook name
    notebook_name = os.path.splitext(os.path.basename(recipe))[0]

    # Read the recipe file
    with open(recipe) as f:
        recipe_data = f.read()

    # Split the recipe into cells and parse each cell
    for cell_data in recipe_data.split("__cellbreak__\n"):
        cell = ""
        cell_type = cell_data.split("\n")[0]

        # Parse each line in the cell
        for cell_line in cell_data.split("\n")[1:]:
            if check_include(cell_line):
                # Handle $include and $tag2tag directives
                if check_tag2tag(cell_line):
                    lines_to_add = swap_tags(
                        get_include(cell_line.split("$include: ")[1].split(" $tag2tag: ")[0], ingredient_path),
                        cell_line.split("$tag2tag: ")[1],
                    )
                    cell += lines_to_add + "\n"
                else:
                    cell += get_include(cell_line.split("$include: ")[1], ingredient_path) + "\n"
            else:
                # Handle $tag2tag directives
                if check_tag2tag(cell_line):
                    tags_split = cell_line.split(" $tag2tag: ")
                    line_to_add = swap_tags(tags_split[0], tags_split[1])
                    cell += line_to_add + "\n"
                else:
                    cell += cell_line + "\n"

        # Create a new cell in the notebook based on the cell type
        if cell_type == "$markdown":
            new_cell = nbformat.v4.new_markdown_cell(cell)
            notebook["cells"].append(new_cell)
        elif cell_type == "$code":
            new_cell = nbformat.v4.new_code_cell(cell)
            new_cell["metadata"]["cellView"] = "form"
            notebook["cells"].append(new_cell)
        else:
            print("ERROR: Unknown cell type: " + cell_type)

    # Write the parsed notebook to the output path
    nbformat.write(notebook, os.path.join(output_path, notebook_name + ".ipynb"))


def compile_notebook(recipe_path: str, ingredient_path: str, output_path: str) -> None:
    """
    Compiles a notebook using nbformat.v4 from a recipe file and ingredients.

    :param recipe_path: Path to the recipe file.
    :type recipe_path: str
    :param ingredient_path: Path to the directory containing the ingredient files.
    :type ingredient_path: str
    :param output_path: Path to write the output notebook file.
    :type output_path: str
    :return: None
    """
    # Create a new notebook object
    notebook = nbformat.v4.new_notebook()

    # Parse the recipe and add the parsed content to the notebook
    parse_recipe(recipe_path, notebook, ingredient_path, output_path)


def compile_all_notebooks(
    recipes_dir: str = "src/notebookchef/recipes", pantry_dir: str = "src/notebookchef/pantry", output_dir: str = "notebooks"
) -> None:
    """Compile all the notebooks located in the recipes folder and store the output in the notebooks folder.

    :param recipes_dir: Path to the folder containing the recipes to compile.
    :param pantry_dir: Path to the folder containing the pantry items to use in the notebooks.
    :param output_dir: Path to the folder where the compiled notebooks will be stored.
    :return: None
    """
    for recipe_file in os.listdir(recipes_dir):
        if recipe_file.endswith(".txt"):
            print("Compiling " + recipe_file)
            recipe_path = os.path.join(recipes_dir, recipe_file)
            compile_notebook(recipe_path, pantry_dir, output_dir)

if __name__ == "__main__":
    compile_all_notebooks()
