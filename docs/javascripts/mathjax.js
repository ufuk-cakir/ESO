window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

if (typeof document$ !== "undefined") {
  document$.subscribe(function () {
    if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
      if (MathJax.typesetClear) MathJax.typesetClear();
      MathJax.typesetPromise();
    }
  });
}
