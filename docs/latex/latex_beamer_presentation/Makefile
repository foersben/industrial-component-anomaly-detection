TEX_FILE = main.tex
PDF_FILE = main.pdf

all: pdf

pdf:
	pdflatex -interaction=nonstopmode $(TEX_FILE)
	pdflatex -interaction=nonstopmode $(TEX_FILE)

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

fclean: clean
	rm -f $(PDF_FILE)

re: fclean all

.PHONY: all pdf clean fclean re
