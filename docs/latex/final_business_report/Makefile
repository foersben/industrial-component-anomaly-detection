.PHONY: all clean

TEX_FILE = report.tex
PDF_FILE = report.pdf

all: $(PDF_FILE)

$(PDF_FILE): $(TEX_FILE) sections/*.tex
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX_FILE)
	# Run a second time to resolve cross-references and page numbers
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX_FILE)

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz sections/*.aux $(PDF_FILE)
