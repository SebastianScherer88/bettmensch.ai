(function (e) {
    function t(t) {
        for (var n, s, r = t[0], c = t[1], l = t[2], d = 0, m = []; d < r.length; d++)
            s = r[d], Object.prototype.hasOwnProperty.call(o, s) && o[s] && m.push(o[s][0]), o[s] = 0;
        for (n in c)
            Object.prototype.hasOwnProperty.call(c, n) && (e[n] = c[n]);
        u && u(t);
        while (m.length)
            m.shift()();
        return i.push.apply(i, l || []),
        a()
    }
    function a() {
        for (var e, t = 0; t < i.length; t++) {
            for (var a = i[t], n = !0, r = 1; r < a.length; r++) {
                var c = a[r];
                0 !== o[c] && (n = !1)
            }
            n && (i.splice(t--, 1), e = s(s.s = a[0]))
        }
        return e
    }
    var n = {},
    o = {
        app: 0
    },
    i = [];
    function s(t) {
        if (n[t])
            return n[t].exports;
        var a = n[t] = {
            i: t,
            l: !1,
            exports: {}
        };
        return e[t].call(a.exports, a, a.exports, s),
        a.l = !0,
        a.exports
    }
    s.m = e,
    s.c = n,
    s.d = function (e, t, a) {
        s.o(e, t) || Object.defineProperty(e, t, {
            enumerable: !0,
            get: a
        })
    },
    s.r = function (e) {
        "undefined" !== typeof Symbol && Symbol.toStringTag && Object.defineProperty(e, Symbol.toStringTag, {
            value: "Module"
        }),
        Object.defineProperty(e, "__esModule", {
            value: !0
        })
    },
    s.t = function (e, t) {
        if (1 & t && (e = s(e)), 8 & t)
            return e;
        if (4 & t && "object" === typeof e && e && e.__esModule)
            return e;
        var a = Object.create(null);
        if (s.r(a), Object.defineProperty(a, "default", {
                enumerable: !0,
                value: e
            }), 2 & t && "string" != typeof e)
            for (var n in e)
                s.d(a, n, function (t) {
                    return e[t]
                }
                    .bind(null, n));
        return a
    },
    s.n = function (e) {
        var t = e && e.__esModule ? function () {
            return e["default"]
        }
         : function () {
            return e
        };
        return s.d(t, "a", t),
        t
    },
    s.o = function (e, t) {
        return Object.prototype.hasOwnProperty.call(e, t)
    },
    s.p = "";
    var r = window["webpackJsonp"] = window["webpackJsonp"] || [],
    c = r.push.bind(r);
    r.push = t,
    r = r.slice();
    for (var l = 0; l < r.length; l++)
        t(r[l]);
    var u = c;
    i.push([0, "chunk-vendors"]),
    a()
})({
    0: function (e, t, a) {
        e.exports = a("cd49")
    },
    "034f": function (e, t, a) {
        "use strict";
        a("85ec")
    },
    "0503": function (e, t, a) {},
    "78ea": function (e, t, a) {
        "use strict";
        a("a8a0")
    },
    "85ec": function (e, t, a) {},
    "96ab": function (e, t, a) {
        "use strict";
        a("0503")
    },
    a347: function (e, t, a) {},
    a8a0: function (e, t, a) {},
    cd49: function (e, t, a) {
        "use strict";
        a.r(t);
        a("e260"),
        a("e6cf"),
        a("cca6"),
        a("a79d");
        var n = a("2b0e"),
        o = function () {
            var e = this,
            t = e.$createElement,
            a = e._self._c || t;
            return a("div", {
                attrs: {
                    id: "app"
                }
            }, [a("WithStreamlitConnection", {
                        scopedSlots: e._u([{
                                    key: "default",
                                    fn: function (e) {
                                        var t = e.args;
                                        return [a("block-editor", {
                                                attrs: {
                                                    args: t
                                                }
                                            })]
                                    }
                                }
                            ])
                    })], 1)
        },
        i = [],
        s = function () {
            var e = this,
            t = e.$createElement,
            a = e._self._c || t;
            return a("div", ["" != e.componentError ? a("div", [a("h1", {
                                staticClass: "err__title"
                            }, [e._v("Component Error")]), a("div", {
                                staticClass: "err__msg"
                            }, [e._v(e._s(e.componentError))])]) : void 0 != e.renderData ? e._t("default", null, {
                        args: e.renderData.args,
                        disabled: e.renderData.disabled
                    }) : e._e()], 2)
        },
        r = [],
        c = a("d092"),
        l = n["default"].extend({
            name: "withStreamlitConnection",
            data: function () {
                return {
                    renderData: void 0,
                    componentError: ""
                }
            },
            methods: {
                onRenderEvent: function (e) {
                    var t = e;
                    this.renderData = t.detail,
                    this.componentError = ""
                }
            },
            mounted: function () {
                c["a"].events.addEventListener(c["a"].RENDER_EVENT, this.onRenderEvent),
                c["a"].setComponentReady(),
                c["a"].setFrameHeight()
            },
            updated: function () {
                c["a"].setFrameHeight()
            },
            destroyed: function () {
                c["a"].events.removeEventListener(c["a"].RENDER_EVENT, this.onRenderEvent)
            },
            errorCaptured: function (e) {
                this.componentError = String(e)
            }
        }),
        u = l,
        d = (a("78ea"), a("2877")),
        m = Object(d["a"])(u, s, r, !1, null, "42636045", null),
        v = m.exports,
        p = function () {
            var e = this,
            t = e.$createElement,
            a = e._self._c || t;
            return a("div", {
                attrs: {
                    id: "editorCanvas"
                }
            }, [a("div", {
                        staticClass: "modal",
                        style: e.menuModal ? "display: block;" : "display: none;"
                    }, [a("div", {
                                staticClass: "modal-content"
                            }, [a("span", {
                                        staticClass: "close",
                                        on: {
                                            click: function (t) {
                                                e.menuModal = !e.menuModal
                                            }
                                        }
                                    }, [e._v("×")]), a("div", {
                                        staticClass: "tab"
                                    }, [a("button", {
                                                staticClass: "tablinks",
                                                class: e.listTab ? "active" : "",
                                                on: {
                                                    click: function (t) {
                                                        return e.activateTab("listTab")
                                                    }
                                                }
                                            }, [e._v(" List ")]), a("button", {
                                                staticClass: "tablinks",
                                                class: e.saveTab ? "active" : "",
                                                on: {
                                                    click: function (t) {
                                                        return e.activateTab("saveTab")
                                                    }
                                                }
                                            }, [e._v(" Save ")])]), a("div", {
                                        staticClass: "tabcontent",
                                        style: e.listTab ? "display: block;" : "display: none;"
                                    }, [a("label", [e._v("List of saved schemas")]), a("ul", e._l(e.loadSchemas, (function (t, n) {
                                                        return a("li", {
                                                            key: n
                                                        }, [e._v(" " + e._s(t) + " ")])
                                                    })), 0), a("p", [e._v(" Current schema: "), a("span", {
                                                        staticStyle: {
                                                            "font-weight": "600"
                                                        }
                                                    }, [e._v(e._s(this.loadSchemaName))])])]), a("div", {
                                        staticClass: "tabcontent",
                                        style: e.saveTab ? "display: block;" : "display: none;"
                                    }, [a("label", [e._v("Enter name to save schema as")]), a("input", {
                                                directives: [{
                                                        name: "model",
                                                        rawName: "v-model",
                                                        value: e.saveSchemaName,
                                                        expression: "saveSchemaName"
                                                    }
                                                ],
                                                attrs: {
                                                    placeholder: "Schema name"
                                                },
                                                domProps: {
                                                    value: e.saveSchemaName
                                                },
                                                on: {
                                                    input: function (t) {
                                                        t.target.composing || (e.saveSchemaName = t.target.value)
                                                    }
                                                }
                                            }), "" !== e.saveSchemaName ? a("button", {
                                                staticClass: "modal-button",
                                                on: {
                                                    click: e.saveEditorData
                                                }
                                            }, [e._v(" Save ")]) : a("button", {
                                                staticClass: "modal-button modal-button-disabled",
                                                on: {
                                                    click: e.saveEditorData
                                                }
                                            }, [e._v(" Save ")]), this.saveSchemaName === this.loadSchemaName ? a("p", [e._v(" The entered schema name is similar to one already existing in the database, saving will override the data for the schema name. ")]) : e._e()])])]), a("baklava-editor", {
                        attrs: {
                            plugin: e.viewPlugin
                        }
                    }), a("div", {
                        staticClass: "button-menu"
                    }, [a("button", {
                                on: {
                                    click: function (t) {
                                        e.menuModal = !e.menuModal
                                    }
                                }
                            }, [e._v("Menu")]), a("button", {
                                on: {
                                    click: e.executeEditorData
                                }
                            }, [e._v("Execute")])])], 1)
        },
        h = [],
        f = (a("d3b7"), a("159b"), a("b0c0"), a("dac5")),
        b = a("0c30"),
        _ = a("9f16"),
        g = a("0048");
        function y(e) {
            var t = e.BlockName,
            a = e.Inputs,
            n = e.Outputs,
            o = e.Options,
            i = new f["NodeBuilder"](t);
            return i.setName(t),
            a.forEach((function (e) {
                    i.addInputInterface(e.name, e.type, e.value, e.properties)
                })),
            n.forEach((function (e) {
                    i.addOutputInterface(e.name, e.type, e.value, e.properties)
                })),
            o.forEach((function (e) {
                    i.addOption(e.name, e.type, e.value, e.sidebar, e.properties)
                })),
            i.build()
        }
        var E = {
            name: "BlockEditor",
            props: ["args"],
            data: function () {
                return {
                    editor: new f["Editor"],
                    viewPlugin: new b["ViewPlugin"],
                    engine: new g["Engine"](!0),
                    menuModal: !1,
                    listTab: !0,
                    saveTab: !1,
                    saveSchemaName: "",
                    loadSchemaName: "",
                    loadSchemas: [],
                    BlockNameID: {}
                }
            },
            created: function () {
                var e = this;
                this.loadSchemas = this.args.load_schema_names,
                this.editor.use(this.viewPlugin),
                this.editor.use(new _["OptionPlugin"]),
                this.editor.use(this.engine),
                this.viewPlugin.enableMinimap = !0,
                console.log(this.args),
                this.args.base_blocks.forEach((function (t) {
                        var a = y({
                            BlockName: t.name,
                            Inputs: t.inputs,
                            Outputs: t.outputs,
                            Options: t.options
                        });
                        Object.prototype.hasOwnProperty.call(t, "category") ? e.editor.registerNodeType(t.name, a, t.category) : e.editor.registerNodeType(t.name, a),
                        e.BlockNameID[t.name] = 1
                    })),
                this.args.load_editor_schema && this.editor.load(this.args.load_editor_schema),
                this.loadSchemaName = this.args.load_schema_name,
                this.editor.events.addNode.addListener(this, (function (t) {
                        e.editor._nodes.forEach((function (a) {
                                a.id === t.id && (a.name = a.name + "-" + e.BlockNameID[t.name]++)
                            }))
                    }))
            },
            methods: {
                executeEditorData: function () {
                    c["a"].setComponentValue({
                        command: "execute",
                        editor_state: this.editor.save()
                    })
                },
                saveEditorData: function () {
                    c["a"].setComponentValue({
                        command: "save",
                        schema_name: this.saveSchemaName,
                        editor_state: this.editor.save()
                    }),
                    this.saveSchemaName = "",
                    this.menuModal = !this.menuModal
                },
                activateTab: function (e) {
                    "listTab" === e && (this.listTab = !0, this.saveTab = !1),
                    "saveTab" === e && (this.listTab = !1, this.saveTab = !0)
                }
            }
        },
        S = E,
        k = (a("96ab"), Object(d["a"])(S, p, h, !1, null, null, null)),
        N = k.exports,
        w = {
            name: "App",
            components: {
                WithStreamlitConnection: v,
                BlockEditor: N
            }
        },
        T = w,
        C = (a("034f"), Object(d["a"])(T, o, i, !1, null, null, null)),
        O = C.exports;
        a("a347");
        n["default"].use(b["BaklavaVuePlugin"]),
        n["default"].config.productionTip = !1,
        n["default"].config.devtools = !1,
        n["default"].prototype.log = console.log,
        new n["default"]({
            render: function (e) {
                return e(O)
            }
        }).$mount("#app")
    }
});
//# sourceMappingURL=app.4b1442b7.js.map
