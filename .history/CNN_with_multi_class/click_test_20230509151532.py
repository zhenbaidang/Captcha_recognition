import click

# 定义命令行命令和参数
@click.command()
@click.option('--name', prompt='Your name', help='Enter your name')
@click.option('--age', prompt='Your age', type=int, help='Enter your age')
def greet(name, age):
    """A command line tool to greet the user."""
    click.echo(f"Hello, {name}! You are {age} years old.")

# 执行命令行命令
if __name__ == '__main__':
    greet()
