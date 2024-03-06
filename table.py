import sys
import os

#if __name__=='__main__':
def write_table(values, names, dataset):

    #values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    #names = ["aug", "info", "reg", "free"]
    #dataset="fern"

    original_stdout = sys.stdout

    ### write latex tabel

    if not os.path.exists(f'tables/latex/{dataset}_table.txt'):
        open(f'tables/latex/{dataset}_table.txt', 'x')

    dat=[]
    with open(f'tables/latex/{dataset}_table.txt', 'r+') as f:
        sys.stdout=f
        dat = f.readlines()[:-2]

        if(len(dat)==0):
            dat.append("\\rowcolors{2}{LightCyan}{White}\n")
            dat.append("\\begin{tabular}{c | c c c | c}\n")
            dat.append("\\hline\n")
            dat.append(f"Scene: {dataset} & PSNR $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ & Average $\\downarrow$  \\\\\n")
            dat.append("\\hline\n")


    with open(f'tables/latex/{dataset}_table.txt', 'w+') as f:
        sys.stdout=f

        for line in dat:
            print(line, end="")

        for i in range(len(names)):
            avg = (10**(-values[i*3]/10) * (1-values[i*3+2])**(1/2) * values[i*3+1])**(1/3)
            print(f"{names[i]} & {values[i*3]:.2f} & {values[i*3 +2]:.3f} & {values[i*3 +1]:.3f} & {avg:.3f}     \\\\")

        print("\\hline")
        print("\\end{tabular}")


    ### write only results into file

    with open(f'tables/{dataset}.txt', 'a+') as f:
        sys.stdout=f

        for i in range(len(names)):
            avg = (10**(-values[i*3]/10) * (1-values[i*3+2])**(1/2) * values[i*3+1])**(1/3)
            print(f"{dataset},{names[i]},{values[i*3]:.2f},{values[i*3 +2]:.3f},{values[i*3 +1]:.3f},{avg:.3f}")

    sys.stdout = original_stdout