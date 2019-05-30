import random, click
from pathlib import Path

def compose_script(r):
    test = r.parent/"test"
    validation = r.parent/"validation"
    for d in r.iterdir():
        if d.is_dir():
            print(f"echo \"Lookin into {str(d)}\"")
            all_image_paths = list(d.glob('*'))
            all_image_paths = [str(path) for path in all_image_paths if path.is_file() and path.suffix != '']
            random.shuffle(all_image_paths)
            five_percent = int(len(all_image_paths)*0.1)
            print(f"# len(all_image_paths)={len(all_image_paths)} 5% = {five_percent}")
            val_dest = validation/d.name
            print(f"echo \"Copy to validation set\"; mkdir -p {val_dest}")
            for val in all_image_paths[:five_percent]:
                print(f"mv -vn \"{val}\" \"{val_dest}/\"")

            test_dest = test/d.name
            print(f"echo \"Copy to test set\"; mkdir -p {test_dest}")
            for tes in all_image_paths[five_percent:2*five_percent]:
                print(f"mv -vn \"{tes}\" \"{test_dest}/\"")

@click.command()
@click.argument('root', type=click.Path(exists=True), default="./train")
def main(root):
    compose_script(Path(root))

if __name__ == "__main__":
    main()
