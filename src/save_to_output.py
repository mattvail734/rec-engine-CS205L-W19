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
