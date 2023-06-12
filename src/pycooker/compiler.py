import os
import nbformat


def swap_tags(data, tags):    
    split_tags = tags.split(";")
    
    if len(split_tags[0].split(",")) > 1:
        for tags in split_tags:
            original = tags.split(",")[0]
            new = tags.split(",")[1]
            data = data.replace(original, new)
    
    return data

def check_tag2tag(cell_line):
    if len(cell_line.split("$tag2tag: ")) > 1:
        return True
    return False


def get_include(tag, ingredient_path):
    with open(os.path.join(ingredient_path, tag)) as f:
        tmp = f.read()
        return tmp


def check_include(cell_line):
    if cell_line.startswith('$include: '):
        return True
    return False


def parse_recipe(recipe, notebook, ingredient_path, output_path):
    notebook["cells"] = []
    notebook_name = recipe.split(os.sep)[-1].split(".")[0]
    with open(recipe) as f:
        recipe_data = f.read()

    for cell_data in recipe_data.split("__cellbreak__\n"):
        cell = ""
        cell_type = cell_data.split("\n")[0]
        for cell_line in cell_data.split("\n")[1:]:
            if check_include(cell_line):
                if check_tag2tag(cell_line):
                    lines_to_add = swap_tags(get_include(cell_line.split("$include: ")[1].split(" $tag2tag: ")[0], ingredient_path), cell_line.split("$tag2tag: ")[1])
                    cell += lines_to_add + "\n"
                else:
                    cell += get_include(cell_line.split("$include: ")[1], ingredient_path) + "\n"
            else:
                if check_tag2tag(cell_line):
                    tags_split = cell_line.split(" $tag2tag: ")
                    print(tags_split)
                    line_to_add = swap_tags(tags_split[0], tags_split[1])
                    cell += line_to_add + "\n"
                else:
                    cell += cell_line + "\n"

        if cell_type == "$markdown":
            new_cell = nbformat.v4.new_markdown_cell(cell)
            notebook["cells"].append(new_cell)
        elif cell_type == "$code":
            new_cell = nbformat.v4.new_code_cell(cell)
            new_cell["metadata"]["cellView"] = "form"
            notebook["cells"].append(new_cell)

    nbformat.write(notebook, os.path.join(output_path, notebook_name+".ipynb"))


def compile_notebook(recipe, ingredient_path, output_path):
    notebook = nbformat.v4.new_notebook()
    parse_recipe(recipe, notebook, ingredient_path, output_path)


def compile_all_notebooks(recipes_path=os.path.join("src", "pycooker", "recipes"), ingredient_path=os.path.join("src", "pycooker", "pantry"), output_path=os.path.join("notebooks")):
    for recipe in os.listdir(recipes_path):
        if recipe.endswith('.txt'):
            compile_notebook(os.path.join(recipes_path, recipe), ingredient_path, output_path)
