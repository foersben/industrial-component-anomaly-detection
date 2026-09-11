.PHONY: all clean

TEX_FILE = report.tex
PDF_FILE = report.pdf

all: $(PDF_FILE)

$(PDF_FILE): $(TEX_FILE)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX_FILE)
	# Run a second time to resolve references
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX_FILE)

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
