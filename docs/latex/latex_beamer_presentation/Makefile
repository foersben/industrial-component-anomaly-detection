TEX       = pdflatex -interaction=nonstopmode
MAIN      = main
NOTES     = notes
MAIN_PDF  = $(MAIN).pdf
NOTES_PDF = $(NOTES).pdf

# -----------------------------------------------------------------------
# Default: build both the presentation and the presenter-notes PDF
# -----------------------------------------------------------------------
all: $(MAIN_PDF) $(NOTES_PDF)

# Presentation (two passes for cross-references / ToC)
$(MAIN_PDF): $(MAIN).tex $(wildcard figures/**/*.tex figures/**/*.png)
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex

# Notes-only PDF (two passes)
$(NOTES_PDF): $(NOTES).tex $(MAIN).tex $(wildcard figures/**/*.tex figures/**/*.png)
	$(TEX) $(NOTES).tex
	$(TEX) $(NOTES).tex

# Convenience aliases
presentation: $(MAIN_PDF)
notes:        $(NOTES_PDF)

# -----------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------
clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

fclean: clean
	rm -f $(MAIN_PDF) $(NOTES_PDF)

re: fclean all

.PHONY: all presentation notes clean fclean re
