!function() {
  "use strict";
  (t => {
    const {
      screen: { width: e, height: a },
      navigator: { language: r },
      location: n,
      localStorage: i,
      document: c,
      history: o
    } = t;
    const { hostname: s, href: u } = n;
    const { currentScript: l, referrer: d } = c;

    if (!l) return;

    const m = "data-",
      f = l.getAttribute.bind(l),
      h = f(m + "website-id"),
      p = f(m + "host-url"),
      g = f(m + "tag"),
      y = "false" !== f(m + "auto-track"),
      b = "true" === f(m + "exclude-search"),
      v = f(m + "domains") || "",
      w = v.split(",").map(t => t.trim()),
      S = `${(p || "https://api-gateway.umami.dev" || l.src.split("/").slice(0, -1).join("/")).replace(/\/$/, "")}/api/send`,
      N = `${e}x${a}`,
      T = /data-umami-event-([\w-_]+)/,
      A = m + "umami-event",
      x = 300;

    const O = t => {
      if (t) {
        try {
          const e = decodeURI(t);
          if (e !== t) return e;
        } catch (e) {
          return t;
        }
        return encodeURI(t);
      }
    };

    const U = t => {
      try {
        const { pathname: e, search: a } = new URL(t);
        t = e + a;
      } catch (t) { }
      return b ? t.split("?")[0] : t;
    };

    const j = () => ({
      website: h,
      hostname: s,
      screen: N,
      language: r,
      title: O(q),
      url: O(D),
      referrer: O(_),
      tag: g || void 0
    });

    const k = (t, e, a) => {
      a && (_ = D, D = U(a.toString()), D !== _ && setTimeout(I, x));
    };

    const E = () => !h || i && i.getItem("umami.disabled") || v && !w.includes(s);

    const L = async (t, e = "event") => {
      if (E()) return;
      const a = { "Content-Type": "application/json" };
      void 0 !== R && (a["x-umami-cache"] = R);
      try {
        const r = await fetch(S, {
          method: "POST",
          body: JSON.stringify({ type: e, payload: t }),
          headers: a
        });
        const n = await r.text();
        return R = n;
      } catch (t) { }
    };

    const $ = () => {
      B || (
        I(),
        (() => {
          const t = (t, e, a) => {
            const r = t[e];
            return (...e) => (a.apply(null, e), r.apply(t, e));
          };
          o.pushState = t(o, "pushState", k);
          o.replaceState = t(o, "replaceState", k);
        })(),
        (() => {
          const t = new MutationObserver(([t]) => {
            q = t && t.target ? t.target.text : void 0;
          });
          const e = c.querySelector("head > title");
          e && t.observe(e, { subtree: !0, characterData: !0, childList: !0 });
        })(),
        c.addEventListener("click", async t => {
          const e = t => ["BUTTON", "A"].includes(t),
            a = async t => {
              const e = t.getAttribute.bind(t),
                a = e(A);
              if (a) {
                const r = {};
                return t.getAttributeNames().forEach(t => {
                  const a = t.match(T);
                  a && (r[a[1]] = e(t));
                }), I(a, r);
              }
            },
            r = t.target,
            i = e(r.tagName) ? r : ((t, a) => {
              let r = t;
              for (let t = 0; t < a; t++) {
                if (e(r.tagName)) return r;
                if (r = r.parentElement, !r) return null;
              }
            })(r, 10);
          if (!i) return a(r);
          {
            const { href: e, target: r } = i,
              c = i.getAttribute(A);
            if (c)
              if ("A" === i.tagName) {
                const o = "_blank" === r || t.ctrlKey || t.shiftKey || t.metaKey || t.button && 1 === t.button;
                if (c && e) return o || t.preventDefault(), a(i).then(() => {
                  o || (n.href = e);
                });
              } else if ("BUTTON" === i.tagName) return a(i);
          }
        }, !0),
        B = !0
      );
    };

    const I = (t, e) => L(
      "string" == typeof t ? { ...j(), name: t, data: "object" == typeof e ? e : void 0 } : "object" == typeof t ? t : "function" == typeof t ? t(j()) : j()
    );

    const K = t => L({ ...j(), data: t }, "identify");

    t.umami || (t.umami = { track: I, identify: K });

    let R, B, D = U(u), _ = d !== s ? d : "", q = c.title;

    y && !E() && ("complete" === c.readyState ? $() : c.addEventListener("readystatechange", $, !0));
  })(window);
}();
