/* Shared, optional interface polish: section reveals and the single sliding navigation underline.
 *
 * Everything here is a progressive enhancement. The `js` class below is what lets the stylesheet
 * hide a reveal in the first place, so if this file fails to load, fails to parse, or is blocked,
 * no element is ever left invisible and the current page keeps its plain static underline.
 */
(function () {
  "use strict";

  /* set before the body paints, so content is never visible and then hidden */
  document.documentElement.classList.add("js");

  /* The hero entrance is a CSS animation and needs nothing here. This is only the below-the-fold
     reveal, which does need to know when an element comes into view. */
  function reveals() {
    var items = [].slice.call(document.querySelectorAll(".reveal"));
    var show = function (el) { el.classList.add("is-in"); };

    if (!("IntersectionObserver" in window)) {
      items.forEach(show);                      /* no observer: everything is simply visible */
      return;
    }
    /* revealed once, then unobserved, so scrolling back up does not replay anything */
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { show(e.target); io.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -10% 0px", threshold: 0.01 });
    items.forEach(function (el) { io.observe(el); });
  }

  function navIndicator() {
    var list = document.querySelector(".nav-links");
    var bar = document.querySelector(".nav-indicator");
    if (!list || !bar) { return; }

    var links = [].slice.call(list.querySelectorAll("a"));
    var current = list.querySelector('a[aria-current="page"]') || links[0];
    if (!current) { return; }

    /* Measured each time, so the bar stays aligned after a resize, a font swap or a zoom. */
    function moveTo(link) {
      var box = link.getBoundingClientRect();
      var base = bar.parentNode.getBoundingClientRect();
      bar.style.setProperty("--ind-w", box.width + "px");
      bar.style.setProperty("--ind-x", (box.left - base.left) + "px");
    }
    function rest() { moveTo(current); }

    links.forEach(function (a) {
      a.addEventListener("pointerenter", function () { moveTo(a); });
      a.addEventListener("focus", function () { moveTo(a); });
    });
    list.addEventListener("pointerleave", rest);
    list.addEventListener("focusout", rest);
    window.addEventListener("resize", rest);
    if (document.fonts && document.fonts.ready) { document.fonts.ready.then(rest); }
    /* Coming back through the bfcache skips every load event, and the viewport may have changed
       while the page was away, so the bar is re-measured rather than left where it was. */
    window.addEventListener("pageshow", function (e) { if (e.persisted) { rest(); } });

    rest();                                     /* placed before it is shown, so it does not slide in */
    bar.hidden = false;
  }

  function init() {
    try {
      reveals();
    } catch (e) {
      /* dropping the class makes the stylesheet stop hiding reveals, so nothing stays invisible */
      document.documentElement.classList.remove("js");
    }
    try { navIndicator(); } catch (e) { /* the static underline remains */ }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
