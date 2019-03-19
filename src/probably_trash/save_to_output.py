import os


def store_dataframe(df, file_name):
    dirname = os.path.dirname(__file__)
    output_dir = os.path.relpath('../output', dirname)
    output_path = os.path.join(output_dir, file_name)
    df.to_csv(output_path, index=True, encoding='utf-8')


def save_fig(plt, file_name):
    dirname = os.path.dirname(__file__)
    output_dir = os.path.relpath('../output', dirname)
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path)


def move_to_output(file_name):
    current_dir = os.path.dirname(__file__)
    current_path = os.path.join(current_dir, file_name)
    output_dir = os.path.relpath('../output', current_dir)
    output_path = os.path.join(output_dir, file_name)
    os.rename(current_path, output_path)
