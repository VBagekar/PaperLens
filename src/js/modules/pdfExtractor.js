/* =========================================================
   pdfExtractor.js — PDF parsing using PDF.js (browser-native)
   Module: Extracts raw text from a PDF File object.
   PaperLens v1.0.0
   ========================================================= */

'use strict';

/**
 * @module PdfExtractor
 * Wraps PDF.js to extract text content from a PDF File.
 * All processing is done in the browser — no data leaves the device.
 */
const PdfExtractor = (() => {

  // Configure PDF.js worker (CDN, no user data sent)
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

  /**
   * Extract all text content from a PDF File object.
   *
   * @param {File} file - The PDF file to parse.
   * @param {function(number, number): void} [onPageProgress] - Called after each page (current, total).
   * @returns {Promise<{ text: string, pageCount: number }>}
   */
  async function extract(file, onPageProgress = null) {
    if (!(file instanceof File)) {
      throw new TypeError('PdfExtractor.extract expects a File object.');
    }

    const arrayBuffer = await file.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
    const pdf = await loadingTask.promise;

    const pageCount = pdf.numPages;
    const pageTexts = [];

    for (let i = 1; i <= pageCount; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      const pageText = content.items.map(item => item.str).join(' ');
      pageTexts.push(pageText);

      if (typeof onPageProgress === 'function') {
        onPageProgress(i, pageCount);
      }
    }

    const text = pageTexts.join('\n\n').trim();

    return { text, pageCount };
  }

  return { extract };

})();
