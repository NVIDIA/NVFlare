"use strict";(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[640],{36578:function(e,t,o){var n=this&&this.__createBinding||(Object.create?function(e,t,o,n){void 0===n&&(n=o);var r=Object.getOwnPropertyDescriptor(t,o);(!r||("get"in r?!t.__esModule:r.writable||r.configurable))&&(r={enumerable:!0,get:function(){return t[o]}}),Object.defineProperty(e,n,r)}:function(e,t,o,n){void 0===n&&(n=o),e[n]=t[o]}),r=this&&this.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),i=this&&this.__importStar||function(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var o in e)"default"!==o&&Object.prototype.hasOwnProperty.call(e,o)&&n(t,e,o);return r(t,e),t},a=this&&this.__importDefault||function(e){return e&&e.__esModule?e:{default:e}};Object.defineProperty(t,"__esModule",{value:!0}),t.Styles=void 0;let u=i(o(67294)),c=i(o(85444)),l=o(7347),s=a(o(5801)),f=a(o(90878)),p=a(o(57299)),d=a(o(93967)),h=a(o(68905));t.Styles=h.default;let m={critical:"ActionsCircleClose",info:"StatusCircleInformation",success:"StatusCircleCheck1",warning:"StatusWarning",passive:"StatusWarning"},v=c.default.div`
  ${e=>e.elevation&&"none"!==e.elevation&&`box-shadow: ${e.theme.elevation[e.elevation]};`}
  ${e=>h.default.banner(e.theme)}
`;v.displayName="Banner";let g=(0,c.default)(p.default)`
  ${e=>h.default.bannerIcon(e.theme)};
`;g.displayName="BannerIcon";let b=(0,c.default)(s.default)`
  ${e=>h.default.bannerCloseButton(e.theme)};
`;b.displayName="BannerCloseButton",t.default=function({children:e,className:t,elevation:o="none",icon:n,onClose:r,rounded:i=!1,status:a="info",title:s}){var p;let h=(0,u.useContext)(l.KaizenThemeContext),y=(0,d.default)(t,a,{rounded:i}),w=h.colors.banner.foreground["passive"===a?"passive":"default"],x=null!==(p=null==n?void 0:n.name)&&void 0!==p?p:m[null!=a?a:"info"],E=w;return(null==n?void 0:n.color)!==void 0&&n.color&&(E=h.colors.banner.icon[n.color]),u.default.createElement(c.ThemeProvider,{theme:h},u.default.createElement(v,{className:y,"data-testid":"kui-banner",elevation:o,rounded:i,status:a},u.default.createElement(g,{className:(0,d.default)("banner-icon",null==n?void 0:n.className),name:x,color:E,variant:null==n?void 0:n.variant,size:(null==n?void 0:n.size)||"medium"}),u.default.createElement("div",{className:"content"},s&&u.default.createElement(f.default,{className:"title",textStyle:"h2",color:w},s),u.default.createElement(f.default,{className:"message",textStyle:"p2",color:w},e)),r&&u.default.createElement(b,{icon:{name:"ActionsClose",variant:"solid",color:w},variant:"link",onClick:r})))}},68905:function(e,t){Object.defineProperty(t,"__esModule",{value:!0}),t.default={banner:e=>`
  position: relative;
  display: flex;
  padding: ${e.spacing.four};
  
  align-content: center;
  justify-content: flex-start;
  background-color: inherit;

  color: ${e.colors.banner.foreground.default};
  ${e.mixins.bodySmall}

  &.rounded {
    border-radius: 5px;
  }

  .banner-icon {
    margin-right: ${e.spacing.four};
    flex-shrink: 0;
  }

  .title {
    color: ${e.colors.banner.foreground};
    margin-bottom: ${e.spacing.one};
  }

  .content {
    display: flex;
    flex-direction: column;
    justify-content: center;
  }

  &.passive {
    background: ${e.colors.banner.background.passive};
    color: ${e.colors.banner.foreground.passive};
  }

  &.critical {
    background: linear-gradient(0.25turn, ${e.colors.banner.background.critical.from}, ${e.colors.banner.background.critical.to});
  }
  &.warning {
    background: linear-gradient(0.25turn, ${e.colors.banner.background.warning.from}, ${e.colors.banner.background.warning.to});
  }

  &.success {
    background: linear-gradient(0.25turn, ${e.colors.banner.background.success.from}, ${e.colors.banner.background.success.to});
  }

  &.info {
    background: linear-gradient(0.25turn, ${e.colors.banner.background.info.from}, ${e.colors.banner.background.info.to});
  }
`,bannerIcon:e=>`
  margin-right: ${e.spacing.two};
`,bannerCloseButton:e=>`
  position: absolute;
  top: ${e.spacing.two};
  right: 0;
  margin-right: ${e.spacing.two};
`}},43199:function(e,t,o){var n=this&&this.__importDefault||function(e){return e&&e.__esModule?e:{default:e}};Object.defineProperty(t,"__esModule",{value:!0});let r=n(o(67294)),i=n(o(85444)),a=i.default.div`
  display: inline-flex;
  &::before {
    content: '$';
    margin-right: 0.5rem;
  }
`;a.displayName="NoteCommand",t.default=({className:e,children:t})=>r.default.createElement(a,{className:e},t)},7731:function(e,t,o){var n=this&&this.__createBinding||(Object.create?function(e,t,o,n){void 0===n&&(n=o);var r=Object.getOwnPropertyDescriptor(t,o);(!r||("get"in r?!t.__esModule:r.writable||r.configurable))&&(r={enumerable:!0,get:function(){return t[o]}}),Object.defineProperty(e,n,r)}:function(e,t,o,n){void 0===n&&(n=o),e[n]=t[o]}),r=this&&this.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),i=this&&this.__importStar||function(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var o in e)"default"!==o&&Object.prototype.hasOwnProperty.call(e,o)&&n(t,e,o);return r(t,e),t},a=this&&this.__importDefault||function(e){return e&&e.__esModule?e:{default:e}};Object.defineProperty(t,"__esModule",{value:!0}),t.Styles=t.NoteCommand=void 0;let u=i(o(67294)),c=a(o(93967)),l=i(o(85444)),s=o(7347),f=a(o(5801)),p=a(o(3535)),d=a(o(25394)),h=a(o(75839));t.Styles=h.default;let m=a(o(43199));t.NoteCommand=m.default;let{CopyOnClick:v}=d.default,g=l.default.div`
  ${e=>h.default.note(e.theme)}
`;g.displayName="Note",t.default=function({className:e,children:t,cli:o=!1,code:n=!1,copyValue:r}){let i=(0,u.useContext)(s.KaizenThemeContext),[a,d]=(0,u.useState)(!1),h=(0,c.default)(e,{cli:o,code:n});return u.default.createElement(l.ThemeProvider,{theme:i},u.default.createElement(g,{className:h,"data-testid":"kui-note"},u.default.createElement("div",{className:"note-command-container"},t),r&&u.default.createElement(p.default,{message:"Copied!",trigger:"manual",closeAfterInterval:2500,visible:a},u.default.createElement(v,{value:r},u.default.createElement(f.default,{icon:{name:"FileCopy"},onClick:()=>{d(!0),setTimeout(()=>{d(!1)},2500)},shape:"square",variant:"link"})))))}},75839:function(e,t){Object.defineProperty(t,"__esModule",{value:!0}),t.default={note:e=>`
  position: relative;
  display: flex;
  justify-content: space-between;
  padding: ${e.spacing.three};

  border-radius: 2px;
  background-color: ${e.colors.note.normal.background};

  font-family: ${e.typography.font.body};
  font-weight: ${e.typography.weight.normal};
  font-size: ${e.typography.size.normal};

  .note-command-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: ${e.spacing.one};
  }

  &.code,
  &.cli {
    ${e.mixins.codeblock}
    overflow-wrap: break-word;
  }

  &.code {
    white-space: pre;
  }

  &.cli {
    color: ${e.colors.note.cli.foreground};
    background: ${e.colors.note.cli.background};
  }
`}},3535:function(e,t,o){var n=this&&this.__createBinding||(Object.create?function(e,t,o,n){void 0===n&&(n=o);var r=Object.getOwnPropertyDescriptor(t,o);(!r||("get"in r?!t.__esModule:r.writable||r.configurable))&&(r={enumerable:!0,get:function(){return t[o]}}),Object.defineProperty(e,n,r)}:function(e,t,o,n){void 0===n&&(n=o),e[n]=t[o]}),r=this&&this.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),i=this&&this.__importStar||function(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var o in e)"default"!==o&&Object.prototype.hasOwnProperty.call(e,o)&&n(t,e,o);return r(t,e),t},a=this&&this.__rest||function(e,t){var o={};for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&0>t.indexOf(n)&&(o[n]=e[n]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var r=0,n=Object.getOwnPropertySymbols(e);r<n.length;r++)0>t.indexOf(n[r])&&Object.prototype.propertyIsEnumerable.call(e,n[r])&&(o[n[r]]=e[n[r]]);return o},u=this&&this.__importDefault||function(e){return e&&e.__esModule?e:{default:e}};Object.defineProperty(t,"__esModule",{value:!0}),t.Styles=void 0;let c=i(o(67294)),l=i(o(85444)),s=o(7347),f=u(o(93967)),p=u(o(7213)),d=i(o(32251));t.Styles=d.default,p.default.displayName="RCTooltipComponent";let h=(0,l.default)(function(e){var{className:t,overlayClassName:o}=e,n=a(e,["className","overlayClassName"]);return c.default.createElement(p.default,Object.assign({overlayClassName:(0,f.default)(o,t)},n))})`
  ${e=>(0,d.forwardedTooltip)(e.theme)}
`;h.displayName="RCTooltip",t.default=function({children:e,className:t,closeAfterInterval:o=1e3,header:n,message:r,overlayClassName:i,placement:a="top",trigger:u="hover",status:p="normal",visible:d=!1}){var m,v;let g=(0,c.useContext)(s.KaizenThemeContext),[b,y]=(0,c.useState)(d),[w,x]=(0,c.useState)(null),E=(0,c.useRef)(null);(0,c.useEffect)(()=>()=>{w&&clearTimeout(w)},[w]),(0,c.useEffect)(()=>{y(d)},[d]);let k=(0,c.useCallback)(()=>{"manual"!==u&&y(!1)},[u]);m="mouseleave",v=()=>E.current,(0,c.useEffect)(()=>{let e=v();if(e)return e.addEventListener(m,k),()=>{e.removeEventListener(m,k)}},[m,k,v]);let C=()=>{"manual"!==u&&y(!0)};return c.default.createElement(l.ThemeProvider,{theme:g},c.default.createElement("span",{className:t},c.default.createElement(h,{placement:a,visible:b,overlayClassName:(0,f.default)("ui","tooltip-overlay",i,p),overlay:c.default.createElement("div",{className:"tooltip-message","data-testid":"kui-tooltip"},n&&c.default.createElement("p",{className:"tooltip-header"},n),r)},c.default.createElement("span",{ref:E,className:"tooltip-link",role:"tooltip",onClick:"click"===u?()=>{y(!0),x(window.setTimeout(()=>{y(!1)},o))}:()=>{},onMouseEnter:"hover"===u?C:()=>{},onFocus:"focus"===u?C:()=>{},onBlur:"focus"===u?k:()=>{}},e))))}},32251:function(e,t){Object.defineProperty(t,"__esModule",{value:!0}),t.forwardedTooltip=void 0,t.forwardedTooltip=e=>`
  position: absolute;
  z-index: 70;
  display: block;
  visibility: visible;
  font-size: 12px;
  line-height: 1.5;
  opacity: 1;
  visibility: visible;
  transition: 0.2s opacity ease-out, 0.2s visibility ease-out;

  &.rc-tooltip-hidden {
    visibility: hidden;
    opacity: 0;
  }

  .rc-tooltip-arrow {
    border-color: transparent;
    position: absolute;
    width: 0;
    height: 0;
    border-color: transparent;
    border-style: solid;

      &:after, &:before {
        border: solid transparent;
        content: " ";
        height: 0;
        width: 0;
        position: absolute;
        pointer-events: none;
      }

      &:after {
        border-color: rgba(136, 183, 213, 0);
      }

      &:before {
        border-color: rgba(194, 225, 245, 0);
      }
    }

  &.rc-tooltip-placement-top {
    padding: 5px 0 9px 0;

    .rc-tooltip-arrow {
      left: 50%;
      bottom: 4px;
      margin-left: -5px;
      border-width: 5px 5px 0;
      margin-bottom: 1px;

      &:after, &:before {
        top: 100%;
        left: 50%;
        margin-top: -0.3rem;
      }

      &:after {
        border-top-color: ${e.colors.tooltip.background};
        border-width: 0.5rem;
        margin-left: -0.5rem;
      }

      &:before {
        border-width: 0.625rem;
        margin-left: -0.625rem;
      }
    }
  }

  &.rc-tooltip-placement-bottom {
    padding: 9px 0 5px 0;

    .rc-tooltip-arrow {
      top: 4px;
      left: 50%;
      margin-left: -5px;
      border-width: 0 5px 5px;
      margin-top: 1px;

      &:after, &:before {
        bottom: 0%;
        left: 50%;
        margin-bottom: -0.3rem;
      }

      &:after {
        border-bottom-color: ${e.colors.tooltip.background};
        border-width: 0.5rem;
        margin-left: -0.5rem;
      }

      &:before {
        border-width: 0.625rem;
        margin-left: -0.625rem;
      }
    }
  }

  &.rc-tooltip-placement-right {
    padding: 0 5px 0 9px;

    .rc-tooltip-arrow {
      left: 4px;
      top: 50%;
      margin-top: -5px;
      border-width: 5px 5px 5px 0;
      margin-left: 1px;

      &:after, &:before {
        top: 50%;
        right: 100%;
        margin-right: -0.32rem;
      }

      &:after {
        border-right-color: ${e.colors.tooltip.background};
        border-width: 0.5rem;
        margin-top: -0.5rem;
      }

      &:before {
        border-width: 0.625rem;
        margin-top: -0.625rem;
      }
    }
  }

  &.rc-tooltip-placement-left {
    padding: 0 9px 0 5px;

    .rc-tooltip-arrow {
      right: 4px;
      top: 50%;
      margin-top: -5px;
      border-width: 5px 0 5px 5px;
      margin-right: 1px;

      &:after, &:before {
        top: 50%;
        left: 100%;
        margin-left: -0.32rem;
      }

      &:after {
        border-left-color: ${e.colors.tooltip.background};
        border-width: 0.5rem;
        margin-top: -0.5rem;
      }

      &:before {
        border-width: 0.625rem;
        margin-top: -0.625rem;
      }
    }
  }

  .rc-tooltip-inner {
    box-shadow: 0 0.25rem 0.3125rem 0 rgba(0, 0, 0, 0.12), 0 0 0.125rem 0 rgba(0, 0, 0, 0.07);
    max-width: 10rem;
    padding: ${e.spacing.two} ${e.spacing.four};
    border: 1px solid;
    border-radius: 0;
    background: ${e.colors.tooltip.background};

    .tooltip-header {
      font-family: ${e.typography.font.body};
      font-size: ${e.typography.size.small};
      font-weight: ${e.typography.weight.bold};
      line-height: ${e.typography.lineHeight.baseline};
      margin: 0 0 ${e.spacing.two};
      text-align: left;
      text-decoration: none;
      text-transform: none;
    }

    .tooltip-message {
      font-family: ${e.typography.font.body};
      font-size: ${e.typography.size.small};
      font-weight: ${e.typography.weight.normal};
      line-height: 1.125rem;
      text-align: left;
      text-decoration: none;
      text-transform: none;
    }
  }

  &.normal {
    .rc-tooltip-inner {
      border-color: ${e.colors.tooltip.normal.border};

      .tooltip-header,
      .tooltip-message {
        color: ${e.colors.tooltip.normal.foreground};
      }
    }

    &.rc-tooltip-placement-top {
      .rc-tooltip-arrow:before {
        border-top-color: ${e.colors.tooltip.normal.border};
      }
    }

    &.rc-tooltip-placement-bottom {
      .rc-tooltip-arrow:before {
        border-bottom-color: ${e.colors.tooltip.normal.border};
      }
    }

    &.rc-tooltip-placement-right {
      .rc-tooltip-arrow:before {
        border-right-color: ${e.colors.tooltip.normal.border};
      }
    }

      &.rc-tooltip-placement-left {
      .rc-tooltip-arrow:before {
        border-left-color: ${e.colors.tooltip.normal.border};
      }
    }
  }

  &.success {
    .rc-tooltip-inner {
      border-color: ${e.colors.tooltip.success.border};

      .tooltip-header,
      .tooltip-message {
        color: ${e.colors.tooltip.success.foreground};
      }
    }

    &.rc-tooltip-placement-top {
      .rc-tooltip-arrow:before {
        border-top-color: ${e.colors.tooltip.success.border};
      }
    }

    &.rc-tooltip-placement-bottom {
      .rc-tooltip-arrow:before {
        border-bottom-color: ${e.colors.tooltip.success.border};
      }
    }

    &.rc-tooltip-placement-right {
      .rc-tooltip-arrow:before {
        border-right-color: ${e.colors.tooltip.success.border};
      }
    }

      &.rc-tooltip-placement-left {
      .rc-tooltip-arrow:before {
        border-left-color: ${e.colors.tooltip.success.border};
      }
    }
  }

  &.warning {
    .rc-tooltip-inner {
      border-color: ${e.colors.tooltip.warning.border};

      .tooltip-header,
      .tooltip-message {
        color: ${e.colors.tooltip.warning.foreground};
      }
    }

    &.rc-tooltip-placement-top {
      .rc-tooltip-arrow:before {
        border-top-color: ${e.colors.tooltip.warning.border};
      }
    }

    &.rc-tooltip-placement-bottom {
      .rc-tooltip-arrow:before {
        border-bottom-color: ${e.colors.tooltip.warning.border};
      }
    }

    &.rc-tooltip-placement-right {
      .rc-tooltip-arrow:before {
        border-right-color: ${e.colors.tooltip.warning.border};
      }
    }

      &.rc-tooltip-placement-left {
      .rc-tooltip-arrow:before {
        border-left-color: ${e.colors.tooltip.warning.border};
      }
    }
  }

  &.critical {
    .rc-tooltip-inner {
      border-color: ${e.colors.tooltip.critical.border};

      .tooltip-header,
      .tooltip-message {
        color: ${e.colors.tooltip.critical.foreground};
      }
    }

    &.rc-tooltip-placement-top {
      .rc-tooltip-arrow:before {
        border-top-color: ${e.colors.tooltip.critical.border};
      }
    }

    &.rc-tooltip-placement-bottom {
      .rc-tooltip-arrow:before {
        border-bottom-color: ${e.colors.tooltip.critical.border};
      }
    }

    &.rc-tooltip-placement-right {
      .rc-tooltip-arrow:before {
        border-right-color: ${e.colors.tooltip.critical.border};
      }
    }

      &.rc-tooltip-placement-left {
      .rc-tooltip-arrow:before {
        border-left-color: ${e.colors.tooltip.critical.border};
      }
    }
  }

  &.info {
    .rc-tooltip-inner {
      border-color: ${e.colors.tooltip.info.border};

      .tooltip-header,
      .tooltip-message {
        color: ${e.colors.tooltip.info.foreground};
      }
    }

    &.rc-tooltip-placement-top {
      .rc-tooltip-arrow:before {
        border-top-color: ${e.colors.tooltip.info.border};
      }
    }

    &.rc-tooltip-placement-bottom {
      .rc-tooltip-arrow:before {
        border-bottom-color: ${e.colors.tooltip.info.border};
      }
    }

    &.rc-tooltip-placement-right {
      .rc-tooltip-arrow:before {
        border-right-color: ${e.colors.tooltip.info.border};
      }
    }

      &.rc-tooltip-placement-left {
      .rc-tooltip-arrow:before {
        border-left-color: ${e.colors.tooltip.info.border};
      }
    }
  }
`,t.default={tooltip:e=>`
  .ui.tooltip-overlay {
    ${(0,t.forwardedTooltip)(e)}
  }
`}},7213:function(e,t,o){o.r(t),o.d(t,{Popup:function(){return or},default:function(){return oa}});var n,r,i,a,u,c,l,s,f=o(87462),p=o(47478),d=o(1413),h=o(45987),m=o(67294),v=o(15671),g=o(43144),b=o(97326),y=o(60136),w=o(18486),x=o(4942),E=o(73935),k=function(e){return+setTimeout(e,16)},C=function(e){return clearTimeout(e)};"undefined"!=typeof window&&"requestAnimationFrame"in window&&(k=function(e){return window.requestAnimationFrame(e)},C=function(e){return window.cancelAnimationFrame(e)});var O=0,_=new Map,M=function(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:1,o=O+=1;return!function t(n){if(0===n)_.delete(o),e();else{var r=k(function(){t(n-1)});_.set(o,r)}}(t),o};function T(e,t){if(!e)return!1;if(e.contains)return e.contains(t);for(var o=t;o;){if(o===e)return!0;o=o.parentNode}return!1}function Z(e){return e instanceof HTMLElement||e instanceof SVGElement}function P(e){var t;return(e&&"object"===(0,p.Z)(e)&&Z(e.nativeElement)?e.nativeElement:Z(e)?e:null)||(e instanceof m.Component?null===(t=E.findDOMNode)||void 0===t?void 0:t.call(E,e):null)}M.cancel=function(e){var t=_.get(e);return _.delete(e),C(t)};var S=o(11805),D=function(e,t){"function"==typeof e?e(t):"object"===(0,p.Z)(e)&&e&&"current"in e&&(e.current=t)},N=function(){for(var e=arguments.length,t=Array(e),o=0;o<e;o++)t[o]=arguments[o];var n=t.filter(Boolean);return n.length<=1?n[0]:function(e){t.forEach(function(t){D(t,e)})}},A=function(e){var t,o,n=(0,S.isMemo)(e)?e.type.type:e.type;return("function"!=typeof n||null!==(t=n.prototype)&&void 0!==t&&!!t.render||n.$$typeof===S.ForwardRef)&&("function"!=typeof e||null!==(o=e.prototype)&&void 0!==o&&!!o.render||e.$$typeof===S.ForwardRef)};function j(e,t,o,n){var r=E.unstable_batchedUpdates?function(e){E.unstable_batchedUpdates(o,e)}:o;return null!=e&&e.addEventListener&&e.addEventListener(t,r,n),{remove:function(){null!=e&&e.removeEventListener&&e.removeEventListener(t,r,n)}}}function L(){return!!("undefined"!=typeof window&&window.document&&window.document.createElement)}m.version.split(".")[0];var R=(0,m.forwardRef)(function(e,t){var o=e.didUpdate,n=e.getContainer,r=e.children,i=(0,m.useRef)(),a=(0,m.useRef)();(0,m.useImperativeHandle)(t,function(){return{}});var u=(0,m.useRef)(!1);return!u.current&&L()&&(a.current=n(),i.current=a.current.parentNode,u.current=!0),(0,m.useEffect)(function(){null==o||o(e)}),(0,m.useEffect)(function(){return null===a.current.parentNode&&null!==i.current&&i.current.appendChild(a.current),function(){var e;null===(e=a.current)||void 0===e||null===(e=e.parentNode)||void 0===e||e.removeChild(a.current)}},[]),a.current?E.createPortal(r,a.current):null}),$=o(93967),V=o.n($),H=o(86854),z=function(){if("undefined"==typeof navigator||"undefined"==typeof window)return!1;var e=navigator.userAgent||navigator.vendor||window.opera;return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino|android|ipad|playbook|silk/i.test(e)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw-(n|u)|c55\/|capi|ccwa|cdm-|cell|chtm|cldc|cmd-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc-s|devi|dica|dmob|do(c|p)o|ds(12|-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(-|_)|g1 u|g560|gene|gf-5|g-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd-(m|p|t)|hei-|hi(pt|ta)|hp( i|ip)|hs-c|ht(c(-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i-(20|go|ma)|i230|iac( |-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|-[a-w])|libw|lynx|m1-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|-([1-8]|c))|phil|pire|pl(ay|uc)|pn-2|po(ck|rt|se)|prox|psio|pt-g|qa-a|qc(07|12|21|32|60|-[2-7]|i-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h-|oo|p-)|sdk\/|se(c(-|0|1)|47|mc|nd|ri)|sgh-|shar|sie(-|m)|sk-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h-|v-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl-|tdg-|tel(i|m)|tim-|t-mo|to(pl|sh)|ts(70|m-|m3|m5)|tx-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas-|your|zeto|zte-/i.test(null==e?void 0:e.substr(0,4))},W=m.createContext({}),B=function(e){(0,y.Z)(o,e);var t=(0,w.Z)(o);function o(){return(0,v.Z)(this,o),t.apply(this,arguments)}return(0,g.Z)(o,[{key:"render",value:function(){return this.props.children}}]),o}(m.Component);function F(e){var t=m.useRef();return t.current=e,m.useCallback(function(){for(var e,o=arguments.length,n=Array(o),r=0;r<o;r++)n[r]=arguments[r];return null===(e=t.current)||void 0===e?void 0:e.call.apply(e,[t].concat(n))},[])}var I=L()?m.useLayoutEffect:m.useEffect,Y=function(e,t){var o=m.useRef(!0);I(function(){return e(o.current)},t),I(function(){return o.current=!1,function(){o.current=!0}},[])};function X(e){var t=m.useRef(!1),o=m.useState(e),n=(0,H.Z)(o,2),r=n[0],i=n[1];return m.useEffect(function(){return t.current=!1,function(){t.current=!0}},[]),[r,function(e,o){o&&t.current||i(e)}]}var G={},q=[];function U(e,t){}function K(e,t){}function J(e,t,o){t||G[o]||(e(!1,o),G[o]=!0)}function Q(e,t){J(U,e,t)}Q.preMessage=function(e){q.push(e)},Q.resetWarned=function(){G={}},Q.noteOnce=function(e,t){J(K,e,t)};var ee="none",et="appear",eo="enter",en="leave",er="none",ei="prepare",ea="start",eu="active",ec="prepared";function el(e,t){var o={};return o[e.toLowerCase()]=t.toLowerCase(),o["Webkit".concat(e)]="webkit".concat(t),o["Moz".concat(e)]="moz".concat(t),o["ms".concat(e)]="MS".concat(t),o["O".concat(e)]="o".concat(t.toLowerCase()),o}var es=(n=L(),r="undefined"!=typeof window?window:{},i={animationend:el("Animation","AnimationEnd"),transitionend:el("Transition","TransitionEnd")},!n||("AnimationEvent"in r||delete i.animationend.animation,"TransitionEvent"in r||delete i.transitionend.transition),i),ef={};L()&&(ef=document.createElement("div").style);var ep={};function ed(e){if(ep[e])return ep[e];var t=es[e];if(t)for(var o=Object.keys(t),n=o.length,r=0;r<n;r+=1){var i=o[r];if(Object.prototype.hasOwnProperty.call(t,i)&&i in ef)return ep[e]=t[i],ep[e]}return""}var eh=ed("animationend"),em=ed("transitionend"),ev=!!(eh&&em),eg=eh||"animationend",eb=em||"transitionend";function ey(e,t){return e?"object"===(0,p.Z)(e)?e[t.replace(/-\w/g,function(e){return e[1].toUpperCase()})]:"".concat(e,"-").concat(t):null}var ew=function(e){var t=(0,m.useRef)();function o(t){t&&(t.removeEventListener(eb,e),t.removeEventListener(eg,e))}return m.useEffect(function(){return function(){o(t.current)}},[]),[function(n){t.current&&t.current!==n&&o(t.current),n&&n!==t.current&&(n.addEventListener(eb,e),n.addEventListener(eg,e),t.current=n)},o]},ex=L()?m.useLayoutEffect:m.useEffect,eE=function(){var e=m.useRef(null);function t(){M.cancel(e.current)}return m.useEffect(function(){return function(){t()}},[]),[function o(n){var r=arguments.length>1&&void 0!==arguments[1]?arguments[1]:2;t();var i=M(function(){r<=1?n({isCanceled:function(){return i!==e.current}}):o(n,r-1)});e.current=i},t]},ek=[ei,ea,eu,"end"],eC=[ei,ec];function eO(e){return e===eu||"end"===e}var e_=function(e,t,o){var n=X(er),r=(0,H.Z)(n,2),i=r[0],a=r[1],u=eE(),c=(0,H.Z)(u,2),l=c[0],s=c[1],f=t?eC:ek;return ex(function(){if(i!==er&&"end"!==i){var e=f.indexOf(i),t=f[e+1],n=o(i);!1===n?a(t,!0):t&&l(function(e){function o(){e.isCanceled()||a(t,!0)}!0===n?o():Promise.resolve(n).then(o)})}},[e,i]),m.useEffect(function(){return function(){s()}},[]),[function(){a(ei,!0)},i]},eM=(a=ev,"object"===(0,p.Z)(ev)&&(a=ev.transitionSupport),(u=m.forwardRef(function(e,t){var o=e.visible,n=void 0===o||o,r=e.removeOnLeave,i=void 0===r||r,u=e.forceRender,c=e.children,l=e.motionName,s=e.leavedClassName,f=e.eventProps,p=m.useContext(W).motion,h=!!(e.motionName&&a&&!1!==p),v=(0,m.useRef)(),g=(0,m.useRef)(),b=function(e,t,o,n){var r,i,a,u=n.motionEnter,c=void 0===u||u,l=n.motionAppear,s=void 0===l||l,f=n.motionLeave,p=void 0===f||f,h=n.motionDeadline,v=n.motionLeaveImmediately,g=n.onAppearPrepare,b=n.onEnterPrepare,y=n.onLeavePrepare,w=n.onAppearStart,E=n.onEnterStart,k=n.onLeaveStart,C=n.onAppearActive,O=n.onEnterActive,_=n.onLeaveActive,M=n.onAppearEnd,T=n.onEnterEnd,Z=n.onLeaveEnd,P=n.onVisibleChanged,S=X(),D=(0,H.Z)(S,2),N=D[0],A=D[1],j=(r=m.useReducer(function(e){return e+1},0),i=(0,H.Z)(r,2)[1],a=m.useRef(ee),[F(function(){return a.current}),F(function(e){a.current="function"==typeof e?e(a.current):e,i()})]),L=(0,H.Z)(j,2),R=L[0],$=L[1],V=X(null),z=(0,H.Z)(V,2),W=z[0],B=z[1],I=R(),Y=(0,m.useRef)(!1),G=(0,m.useRef)(null),q=(0,m.useRef)(!1);function U(){$(ee),B(null,!0)}var K=F(function(e){var t,n=R();if(n!==ee){var r=o();if(!e||e.deadline||e.target===r){var i=q.current;n===et&&i?t=null==M?void 0:M(r,e):n===eo&&i?t=null==T?void 0:T(r,e):n===en&&i&&(t=null==Z?void 0:Z(r,e)),i&&!1!==t&&U()}}}),J=ew(K),Q=(0,H.Z)(J,1)[0],er=function(e){switch(e){case et:return(0,x.Z)((0,x.Z)((0,x.Z)({},ei,g),ea,w),eu,C);case eo:return(0,x.Z)((0,x.Z)((0,x.Z)({},ei,b),ea,E),eu,O);case en:return(0,x.Z)((0,x.Z)((0,x.Z)({},ei,y),ea,k),eu,_);default:return{}}},el=m.useMemo(function(){return er(I)},[I]),es=e_(I,!e,function(e){if(e===ei){var t,n=el[ei];return!!n&&n(o())}return ed in el&&B((null===(t=el[ed])||void 0===t?void 0:t.call(el,o(),null))||null),ed===eu&&I!==ee&&(Q(o()),h>0&&(clearTimeout(G.current),G.current=setTimeout(function(){K({deadline:!0})},h))),ed===ec&&U(),!0}),ef=(0,H.Z)(es,2),ep=ef[0],ed=ef[1],eh=eO(ed);q.current=eh,ex(function(){A(t);var o,n=Y.current;Y.current=!0,!n&&t&&s&&(o=et),n&&t&&c&&(o=eo),(n&&!t&&p||!n&&v&&!t&&p)&&(o=en);var r=er(o);o&&(e||r[ei])?($(o),ep()):$(ee)},[t]),(0,m.useEffect)(function(){(I!==et||s)&&(I!==eo||c)&&(I!==en||p)||$(ee)},[s,c,p]),(0,m.useEffect)(function(){return function(){Y.current=!1,clearTimeout(G.current)}},[]);var em=m.useRef(!1);(0,m.useEffect)(function(){N&&(em.current=!0),void 0!==N&&I===ee&&((em.current||N)&&(null==P||P(N)),em.current=!0)},[N,I]);var ev=W;return el[ei]&&ed===ea&&(ev=(0,d.Z)({transition:"none"},ev)),[I,ed,ev,null!=N?N:t]}(h,n,function(){try{return v.current instanceof HTMLElement?v.current:P(g.current)}catch(e){return null}},e),y=(0,H.Z)(b,4),w=y[0],E=y[1],k=y[2],C=y[3],O=m.useRef(C);C&&(O.current=!0);var _=m.useCallback(function(e){v.current=e,D(t,e)},[t]),M=(0,d.Z)((0,d.Z)({},f),{},{visible:n});if(c){if(w===ee)T=C?c((0,d.Z)({},M),_):!i&&O.current&&s?c((0,d.Z)((0,d.Z)({},M),{},{className:s}),_):!u&&(i||s)?null:c((0,d.Z)((0,d.Z)({},M),{},{style:{display:"none"}}),_);else{E===ei?Z="prepare":eO(E)?Z="active":E===ea&&(Z="start");var T,Z,S=ey(l,"".concat(w,"-").concat(Z));T=c((0,d.Z)((0,d.Z)({},M),{},{className:V()(ey(l,w),(0,x.Z)((0,x.Z)({},S,S&&Z),l,"string"==typeof l)),style:k}),_)}}else T=null;return m.isValidElement(T)&&A(T)&&!T.ref&&(T=m.cloneElement(T,{ref:_})),m.createElement(B,{ref:g},T)})).displayName="CSSMotion",u),eT="keep",eZ="remove",eP="removed";function eS(e){var t;return t=e&&"object"===(0,p.Z)(e)&&"key"in e?e:{key:e},(0,d.Z)((0,d.Z)({},t),{},{key:String(t.key)})}function eD(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:[];return e.map(eS)}var eN=["component","children","onVisibleChanged","onAllRemoved"],eA=["status"],ej=["eventProps","visible","children","motionName","motionAppear","motionEnter","motionLeave","motionLeaveImmediately","motionDeadline","removeOnLeave","leavedClassName","onAppearPrepare","onAppearStart","onAppearActive","onAppearEnd","onEnterStart","onEnterActive","onEnterEnd","onLeaveStart","onLeaveActive","onLeaveEnd"];function eL(e){var t=e.prefixCls,o=e.motion,n=e.animation,r=e.transitionName;return o||(n?{motionName:"".concat(t,"-").concat(n)}:r?{motionName:r}:null)}function eR(e){var t=e.prefixCls,o=e.visible,n=e.zIndex,r=e.mask,i=e.maskMotion,a=e.maskAnimation,u=e.maskTransitionName;if(!r)return null;var c={};return(i||u||a)&&(c=(0,d.Z)({motionAppear:!0},eL({motion:i,prefixCls:t,transitionName:u,animation:a}))),m.createElement(eM,(0,f.Z)({},c,{visible:o,removeOnLeave:!0}),function(e){var o=e.className;return m.createElement("div",{style:{zIndex:n},className:V()("".concat(t,"-mask"),o)})})}function e$(e,t){var o=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),o.push.apply(o,n)}return o}function eV(e){for(var t=1;t<arguments.length;t++){var o=null!=arguments[t]?arguments[t]:{};t%2?e$(Object(o),!0).forEach(function(t){var n;n=o[t],t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(o)):e$(Object(o)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(o,t))})}return e}function eH(e){return(eH="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"==typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e})(e)}!function(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:eM,o=function(e){(0,y.Z)(n,e);var o=(0,w.Z)(n);function n(){var e;(0,v.Z)(this,n);for(var t=arguments.length,r=Array(t),i=0;i<t;i++)r[i]=arguments[i];return e=o.call.apply(o,[this].concat(r)),(0,x.Z)((0,b.Z)(e),"state",{keyEntities:[]}),(0,x.Z)((0,b.Z)(e),"removeKey",function(t){var o=e.state.keyEntities.map(function(e){return e.key!==t?e:(0,d.Z)((0,d.Z)({},e),{},{status:eP})});return e.setState({keyEntities:o}),o.filter(function(e){return e.status!==eP}).length}),e}return(0,g.Z)(n,[{key:"render",value:function(){var e=this,o=this.state.keyEntities,n=this.props,r=n.component,i=n.children,a=n.onVisibleChanged,u=n.onAllRemoved,c=(0,h.Z)(n,eN),l=r||m.Fragment,s={};return ej.forEach(function(e){s[e]=c[e],delete c[e]}),delete c.keys,m.createElement(l,c,o.map(function(o,n){var r=o.status,c=(0,h.Z)(o,eA);return m.createElement(t,(0,f.Z)({},s,{key:c.key,visible:"add"===r||r===eT,eventProps:c,onVisibleChanged:function(t){null==a||a(t,{key:c.key}),!t&&0===e.removeKey(c.key)&&u&&u()}}),function(e,t){return i((0,d.Z)((0,d.Z)({},e),{},{index:n}),t)})}))}}],[{key:"getDerivedStateFromProps",value:function(e,t){var o=e.keys,n=t.keyEntities;return{keyEntities:(function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:[],t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:[],o=[],n=0,r=t.length,i=eD(e),a=eD(t);i.forEach(function(e){for(var t=!1,i=n;i<r;i+=1){var u=a[i];if(u.key===e.key){n<i&&(o=o.concat(a.slice(n,i).map(function(e){return(0,d.Z)((0,d.Z)({},e),{},{status:"add"})})),n=i),o.push((0,d.Z)((0,d.Z)({},u),{},{status:eT})),n+=1,t=!0;break}}t||o.push((0,d.Z)((0,d.Z)({},e),{},{status:eZ}))}),n<r&&(o=o.concat(a.slice(n).map(function(e){return(0,d.Z)((0,d.Z)({},e),{},{status:"add"})})));var u={};return o.forEach(function(e){var t=e.key;u[t]=(u[t]||0)+1}),Object.keys(u).filter(function(e){return u[e]>1}).forEach(function(e){(o=o.filter(function(t){var o=t.key,n=t.status;return o!==e||n!==eZ})).forEach(function(t){t.key===e&&(t.status=eT)})}),o})(n,eD(o)).filter(function(e){var t=n.find(function(t){var o=t.key;return e.key===o});return!t||t.status!==eP||e.status!==eZ})}}}]),n}(m.Component);(0,x.Z)(o,"defaultProps",{component:"div"})}(ev);var ez={Webkit:"-webkit-",Moz:"-moz-",ms:"-ms-",O:"-o-"};function eW(){if(void 0!==l)return l;l="";var e=document.createElement("p").style;for(var t in ez)t+"Transform" in e&&(l=t);return l}function eB(){return eW()?"".concat(eW(),"TransitionProperty"):"transitionProperty"}function eF(){return eW()?"".concat(eW(),"Transform"):"transform"}function eI(e,t){var o=eB();o&&(e.style[o]=t,"transitionProperty"!==o&&(e.style.transitionProperty=t))}function eY(e,t){var o=eF();o&&(e.style[o]=t,"transform"!==o&&(e.style.transform=t))}var eX=/matrix\((.*)\)/,eG=/matrix3d\((.*)\)/,eq=/[\-+]?(?:\d*\.|)\d+(?:[eE][\-+]?\d+|)/.source;function eU(e){var t=e.style.display;e.style.display="none",e.offsetHeight,e.style.display=t}function eK(e,t,o){var n=o;if("object"===eH(t)){for(var r in t)t.hasOwnProperty(r)&&eK(e,r,t[r]);return}if(void 0!==n){"number"==typeof n&&(n="".concat(n,"px")),e.style[t]=n;return}return s(e,t)}function eJ(e,t){var o=e["page".concat(t?"Y":"X","Offset")],n="scroll".concat(t?"Top":"Left");if("number"!=typeof o){var r=e.document;"number"!=typeof(o=r.documentElement[n])&&(o=r.body[n])}return o}function eQ(e){var t,o,n,r,i,a,u=(i=(r=e.ownerDocument).body,a=r&&r.documentElement,o=Math.floor((t=e.getBoundingClientRect()).left),n=Math.floor(t.top),{left:o-=a.clientLeft||i.clientLeft||0,top:n-=a.clientTop||i.clientTop||0}),c=e.ownerDocument,l=c.defaultView||c.parentWindow;return u.left+=eJ(l),u.top+=eJ(l,!0),u}function e0(e){return null!=e&&e==e.window}function e1(e){return e0(e)?e.document:9===e.nodeType?e:e.ownerDocument}var e2=RegExp("^(".concat(eq,")(?!px)[a-z%]+$"),"i"),e5=/^(top|right|bottom|left)$/,e4="currentStyle",e3="runtimeStyle",e9="left";function e7(e,t){return"left"===e?t.useCssRight?"right":e:t.useCssBottom?"bottom":e}function e6(e){return"left"===e?"right":"right"===e?"left":"top"===e?"bottom":"bottom"===e?"top":void 0}function e8(e,t,o){"static"===eK(e,"position")&&(e.style.position="relative");var n=-999,r=-999,i=e7("left",o),a=e7("top",o),u=e6(i),c=e6(a);"left"!==i&&(n=999),"top"!==a&&(r=999);var l="",s=eQ(e);("left"in t||"top"in t)&&(l=e.style.transitionProperty||e.style[eB()]||"",eI(e,"none")),"left"in t&&(e.style[u]="",e.style[i]="".concat(n,"px")),"top"in t&&(e.style[c]="",e.style[a]="".concat(r,"px")),eU(e);var f=eQ(e),p={};for(var d in t)if(t.hasOwnProperty(d)){var h=e7(d,o),m="left"===d?n:r,v=s[d]-f[d];h===d?p[h]=m+v:p[h]=m-v}eK(e,p),eU(e),("left"in t||"top"in t)&&eI(e,l);var g={};for(var b in t)if(t.hasOwnProperty(b)){var y=e7(b,o),w=t[b]-s[b];b===y?g[y]=p[y]+w:g[y]=p[y]-w}eK(e,g)}function te(e,t){for(var o=0;o<e.length;o++)t(e[o])}function tt(e){return"border-box"===s(e,"boxSizing")}"undefined"!=typeof window&&(s=window.getComputedStyle?function(e,t,o){var n=o,r="",i=e1(e);return(n=n||i.defaultView.getComputedStyle(e,null))&&(r=n.getPropertyValue(t)||n[t]),r}:function(e,t){var o=e[e4]&&e[e4][t];if(e2.test(o)&&!e5.test(t)){var n=e.style,r=n[e9],i=e[e3][e9];e[e3][e9]=e[e4][e9],n[e9]="fontSize"===t?"1em":o||0,o=n.pixelLeft+"px",n[e9]=r,e[e3][e9]=i}return""===o?"auto":o});var to=["margin","border","padding"];function tn(e,t,o){var n,r,i,a=0;for(r=0;r<t.length;r++)if(n=t[r])for(i=0;i<o.length;i++){var u=void 0;u="border"===n?"".concat(n).concat(o[i],"Width"):n+o[i],a+=parseFloat(s(e,u))||0}return a}var tr={getParent:function(e){var t=e;do t=11===t.nodeType&&t.host?t.host:t.parentNode;while(t&&1!==t.nodeType&&9!==t.nodeType);return t}};function ti(e,t,o){var n=o;if(e0(e))return"width"===t?tr.viewportWidth(e):tr.viewportHeight(e);if(9===e.nodeType)return"width"===t?tr.docWidth(e):tr.docHeight(e);var r="width"===t?["Left","Right"]:["Top","Bottom"],i="width"===t?Math.floor(e.getBoundingClientRect().width):Math.floor(e.getBoundingClientRect().height),a=tt(e),u=0;(null==i||i<=0)&&(i=void 0,(null==(u=s(e,t))||0>Number(u))&&(u=e.style[t]||0),u=Math.floor(parseFloat(u))||0),void 0===n&&(n=a?1:-1);var c=void 0!==i||a,l=i||u;return -1===n?c?l-tn(e,["border","padding"],r):u:c?1===n?l:l+(2===n?-tn(e,["border"],r):tn(e,["margin"],r)):u+tn(e,to.slice(n),r)}te(["Width","Height"],function(e){tr["doc".concat(e)]=function(t){var o=t.document;return Math.max(o.documentElement["scroll".concat(e)],o.body["scroll".concat(e)],tr["viewport".concat(e)](o))},tr["viewport".concat(e)]=function(t){var o="client".concat(e),n=t.document,r=n.body,i=n.documentElement[o];return"CSS1Compat"===n.compatMode&&i||r&&r[o]||i}});var ta={position:"absolute",visibility:"hidden",display:"block"};function tu(){for(var e,t=arguments.length,o=Array(t),n=0;n<t;n++)o[n]=arguments[n];var r=o[0];return 0!==r.offsetWidth?e=ti.apply(void 0,o):function(e,t,o){var n,r={},i=e.style;for(n in t)t.hasOwnProperty(n)&&(r[n]=i[n],i[n]=t[n]);for(n in o.call(e),t)t.hasOwnProperty(n)&&(i[n]=r[n])}(r,ta,function(){e=ti.apply(void 0,o)}),e}function tc(e,t){for(var o in t)t.hasOwnProperty(o)&&(e[o]=t[o]);return e}te(["width","height"],function(e){var t=e.charAt(0).toUpperCase()+e.slice(1);tr["outer".concat(t)]=function(t,o){return t&&tu(t,e,o?0:1)};var o="width"===e?["Left","Right"]:["Top","Bottom"];tr[e]=function(t,n){var r=n;return void 0!==r?t?(tt(t)&&(r+=tn(t,["padding","border"],o)),eK(t,e,r)):void 0:t&&tu(t,e,-1)}});var tl={getWindow:function(e){if(e&&e.document&&e.setTimeout)return e;var t=e.ownerDocument||e;return t.defaultView||t.parentWindow},getDocument:e1,offset:function(e,t,o){if(void 0===t)return eQ(e);!function(e,t,o){if(o.ignoreShake){var n,r,i,a=eQ(e),u=a.left.toFixed(0),c=a.top.toFixed(0),l=t.left.toFixed(0),s=t.top.toFixed(0);if(u===l&&c===s)return}o.useCssRight||o.useCssBottom?e8(e,t,o):o.useCssTransform&&eF() in document.body.style?(n=eQ(e),i={x:(r=function(e){var t=window.getComputedStyle(e,null),o=t.getPropertyValue("transform")||t.getPropertyValue(eF());if(o&&"none"!==o){var n=o.replace(/[^0-9\-.,]/g,"").split(",");return{x:parseFloat(n[12]||n[4],0),y:parseFloat(n[13]||n[5],0)}}return{x:0,y:0}}(e)).x,y:r.y},"left"in t&&(i.x=r.x+t.left-n.left),"top"in t&&(i.y=r.y+t.top-n.top),function(e,t){var o=window.getComputedStyle(e,null),n=o.getPropertyValue("transform")||o.getPropertyValue(eF());if(n&&"none"!==n){var r,i=n.match(eX);i?((r=(i=i[1]).split(",").map(function(e){return parseFloat(e,10)}))[4]=t.x,r[5]=t.y,eY(e,"matrix(".concat(r.join(","),")"))):((r=n.match(eG)[1].split(",").map(function(e){return parseFloat(e,10)}))[12]=t.x,r[13]=t.y,eY(e,"matrix3d(".concat(r.join(","),")")))}else eY(e,"translateX(".concat(t.x,"px) translateY(").concat(t.y,"px) translateZ(0)"))}(e,i)):e8(e,t,o)}(e,t,o||{})},isWindow:e0,each:te,css:eK,clone:function(e){var t,o={};for(t in e)e.hasOwnProperty(t)&&(o[t]=e[t]);if(e.overflow)for(t in e)e.hasOwnProperty(t)&&(o.overflow[t]=e.overflow[t]);return o},mix:tc,getWindowScrollLeft:function(e){return eJ(e)},getWindowScrollTop:function(e){return eJ(e,!0)},merge:function(){for(var e={},t=0;t<arguments.length;t++)tl.mix(e,t<0||arguments.length<=t?void 0:arguments[t]);return e},viewportWidth:0,viewportHeight:0};tc(tl,tr);var ts=tl.getParent;function tf(e){if(tl.isWindow(e)||9===e.nodeType)return null;var t,o=tl.getDocument(e).body,n=tl.css(e,"position");if(!("fixed"===n||"absolute"===n))return"html"===e.nodeName.toLowerCase()?null:ts(e);for(t=ts(e);t&&t!==o&&9!==t.nodeType;t=ts(t))if("static"!==(n=tl.css(t,"position")))return t;return null}var tp=tl.getParent;function td(e,t){for(var o={left:0,right:1/0,top:0,bottom:1/0},n=tf(e),r=tl.getDocument(e),i=r.defaultView||r.parentWindow,a=r.body,u=r.documentElement;n;){if((-1===navigator.userAgent.indexOf("MSIE")||0!==n.clientWidth)&&n!==a&&n!==u&&"visible"!==tl.css(n,"overflow")){var c=tl.offset(n);c.left+=n.clientLeft,c.top+=n.clientTop,o.top=Math.max(o.top,c.top),o.right=Math.min(o.right,c.left+n.clientWidth),o.bottom=Math.min(o.bottom,c.top+n.clientHeight),o.left=Math.max(o.left,c.left)}else if(n===a||n===u)break;n=tf(n)}var l=null;tl.isWindow(e)||9===e.nodeType||(l=e.style.position,"absolute"!==tl.css(e,"position")||(e.style.position="fixed"));var s=tl.getWindowScrollLeft(i),f=tl.getWindowScrollTop(i),p=tl.viewportWidth(i),d=tl.viewportHeight(i),h=u.scrollWidth,m=u.scrollHeight,v=window.getComputedStyle(a);if("hidden"===v.overflowX&&(h=i.innerWidth),"hidden"===v.overflowY&&(m=i.innerHeight),e.style&&(e.style.position=l),t||function(e){if(tl.isWindow(e)||9===e.nodeType)return!1;var t=tl.getDocument(e),o=t.body,n=null;for(n=tp(e);n&&n!==o&&n!==t;n=tp(n))if("fixed"===tl.css(n,"position"))return!0;return!1}(e))o.left=Math.max(o.left,s),o.top=Math.max(o.top,f),o.right=Math.min(o.right,s+p),o.bottom=Math.min(o.bottom,f+d);else{var g=Math.max(h,s+p);o.right=Math.min(o.right,g);var b=Math.max(m,f+d);o.bottom=Math.min(o.bottom,b)}return o.top>=0&&o.left>=0&&o.bottom>o.top&&o.right>o.left?o:null}function th(e){if(tl.isWindow(e)||9===e.nodeType){var t,o,n,r=tl.getWindow(e);t={left:tl.getWindowScrollLeft(r),top:tl.getWindowScrollTop(r)},o=tl.viewportWidth(r),n=tl.viewportHeight(r)}else t=tl.offset(e),o=tl.outerWidth(e),n=tl.outerHeight(e);return t.width=o,t.height=n,t}function tm(e,t){var o=t.charAt(0),n=t.charAt(1),r=e.width,i=e.height,a=e.left,u=e.top;return"c"===o?u+=i/2:"b"===o&&(u+=i),"c"===n?a+=r/2:"r"===n&&(a+=r),{left:a,top:u}}function tv(e,t,o,n,r){var i=tm(t,o[1]),a=tm(e,o[0]),u=[a.left-i.left,a.top-i.top];return{left:Math.round(e.left-u[0]+n[0]-r[0]),top:Math.round(e.top-u[1]+n[1]-r[1])}}function tg(e,t,o){return e.left<o.left||e.left+t.width>o.right}function tb(e,t,o){return e.top<o.top||e.top+t.height>o.bottom}function ty(e,t,o){var n=[];return tl.each(e,function(e){n.push(e.replace(t,function(e){return o[e]}))}),n}function tw(e,t){return e[t]=-e[t],e}function tx(e,t){return(/%$/.test(e)?parseInt(e.substring(0,e.length-1),10)/100*t:parseInt(e,10))||0}function tE(e,t){e[0]=tx(e[0],t.width),e[1]=tx(e[1],t.height)}function tk(e,t,o,n){var r=o.points,i=o.offset||[0,0],a=o.targetOffset||[0,0],u=o.overflow,c=o.source||e;i=[].concat(i),a=[].concat(a);var l={},s=0,f=td(c,!!(u=u||{}).alwaysByViewport),p=th(c);tE(i,p),tE(a,t);var d=tv(p,t,r,i,a),h=tl.merge(p,d);if(f&&(u.adjustX||u.adjustY)&&n){if(u.adjustX&&tg(d,p,f)){var m,v,g,b,y=ty(r,/[lr]/gi,{l:"r",r:"l"}),w=tw(i,0),x=tw(a,0);(b=tv(p,t,y,w,x)).left>f.right||b.left+p.width<f.left||(s=1,r=y,i=w,a=x)}if(u.adjustY&&tb(d,p,f)){var E,k=ty(r,/[tb]/gi,{t:"b",b:"t"}),C=tw(i,1),O=tw(a,1);(E=tv(p,t,k,C,O)).top>f.bottom||E.top+p.height<f.top||(s=1,r=k,i=C,a=O)}s&&(d=tv(p,t,r,i,a),tl.mix(h,d));var _=tg(d,p,f),M=tb(d,p,f);if(_||M){var T=r;_&&(T=ty(r,/[lr]/gi,{l:"r",r:"l"})),M&&(T=ty(r,/[tb]/gi,{t:"b",b:"t"})),r=T,i=o.offset||[0,0],a=o.targetOffset||[0,0]}l.adjustX=u.adjustX&&_,l.adjustY=u.adjustY&&M,(l.adjustX||l.adjustY)&&(m=d,v=tl.clone(m),g={width:p.width,height:p.height},l.adjustX&&v.left<f.left&&(v.left=f.left),l.resizeWidth&&v.left>=f.left&&v.left+g.width>f.right&&(g.width-=v.left+g.width-f.right),l.adjustX&&v.left+g.width>f.right&&(v.left=Math.max(f.right-g.width,f.left)),l.adjustY&&v.top<f.top&&(v.top=f.top),l.resizeHeight&&v.top>=f.top&&v.top+g.height>f.bottom&&(g.height-=v.top+g.height-f.bottom),l.adjustY&&v.top+g.height>f.bottom&&(v.top=Math.max(f.bottom-g.height,f.top)),h=tl.mix(v,g))}return h.width!==p.width&&tl.css(c,"width",tl.width(c)+h.width-p.width),h.height!==p.height&&tl.css(c,"height",tl.height(c)+h.height-p.height),tl.offset(c,{left:h.left,top:h.top},{useCssRight:o.useCssRight,useCssBottom:o.useCssBottom,useCssTransform:o.useCssTransform,ignoreShake:o.ignoreShake}),{points:r,offset:i,targetOffset:a,overflow:l}}function tC(e,t,o){var n,r,i=o.target||t,a=th(i),u=(n=td(i,o.overflow&&o.overflow.alwaysByViewport),r=th(i),!!n&&!(r.left+r.width<=n.left)&&!(r.top+r.height<=n.top)&&!(r.left>=n.right)&&!(r.top>=n.bottom));return tk(e,a,o,u)}tC.__getOffsetParent=tf,tC.__getVisibleRectForElement=td;var tO=function(e,t){var o=arguments.length>2&&void 0!==arguments[2]&&arguments[2],n=new Set;return function e(t,r){var i=arguments.length>2&&void 0!==arguments[2]?arguments[2]:1,a=n.has(t);if(Q(!a,"Warning: There may be circular references"),a)return!1;if(t===r)return!0;if(o&&i>1)return!1;n.add(t);var u=i+1;if(Array.isArray(t)){if(!Array.isArray(r)||t.length!==r.length)return!1;for(var c=0;c<t.length;c++)if(!e(t[c],r[c],u))return!1;return!0}if(t&&r&&"object"===(0,p.Z)(t)&&"object"===(0,p.Z)(r)){var l=Object.keys(t);return l.length===Object.keys(r).length&&l.every(function(o){return e(t[o],r[o],u)})}return!1}(e,t)},t_=function(e){if(!e)return!1;if(e instanceof Element){if(e.offsetParent)return!0;if(e.getBBox){var t=e.getBBox(),o=t.width,n=t.height;if(o||n)return!0}if(e.getBoundingClientRect){var r=e.getBoundingClientRect(),i=r.width,a=r.height;if(i||a)return!0}}return!1},tM=function(e,t){var o=m.useRef(!1),n=m.useRef(null);function r(){window.clearTimeout(n.current)}return[function i(a){if(r(),o.current&&!0!==a)n.current=window.setTimeout(function(){o.current=!1,i()},t);else{if(!1===e(a))return;o.current=!0,n.current=window.setTimeout(function(){o.current=!1},t)}},function(){o.current=!1,r()}]},tT=function(){if("undefined"!=typeof Map)return Map;function e(e,t){var o=-1;return e.some(function(e,n){return e[0]===t&&(o=n,!0)}),o}return function(){function t(){this.__entries__=[]}return Object.defineProperty(t.prototype,"size",{get:function(){return this.__entries__.length},enumerable:!0,configurable:!0}),t.prototype.get=function(t){var o=e(this.__entries__,t),n=this.__entries__[o];return n&&n[1]},t.prototype.set=function(t,o){var n=e(this.__entries__,t);~n?this.__entries__[n][1]=o:this.__entries__.push([t,o])},t.prototype.delete=function(t){var o=this.__entries__,n=e(o,t);~n&&o.splice(n,1)},t.prototype.has=function(t){return!!~e(this.__entries__,t)},t.prototype.clear=function(){this.__entries__.splice(0)},t.prototype.forEach=function(e,t){void 0===t&&(t=null);for(var o=0,n=this.__entries__;o<n.length;o++){var r=n[o];e.call(t,r[1],r[0])}},t}()}(),tZ="undefined"!=typeof window&&"undefined"!=typeof document&&window.document===document,tP=void 0!==o.g&&o.g.Math===Math?o.g:"undefined"!=typeof self&&self.Math===Math?self:"undefined"!=typeof window&&window.Math===Math?window:Function("return this")(),tS="function"==typeof requestAnimationFrame?requestAnimationFrame.bind(tP):function(e){return setTimeout(function(){return e(Date.now())},1e3/60)},tD=["top","right","bottom","left","width","height","size","weight"],tN="undefined"!=typeof MutationObserver,tA=function(){function e(){this.connected_=!1,this.mutationEventsAdded_=!1,this.mutationsObserver_=null,this.observers_=[],this.onTransitionEnd_=this.onTransitionEnd_.bind(this),this.refresh=function(e,t){var o=!1,n=!1,r=0;function i(){o&&(o=!1,e()),n&&u()}function a(){tS(i)}function u(){var e=Date.now();if(o){if(e-r<2)return;n=!0}else o=!0,n=!1,setTimeout(a,20);r=e}return u}(this.refresh.bind(this),0)}return e.prototype.addObserver=function(e){~this.observers_.indexOf(e)||this.observers_.push(e),this.connected_||this.connect_()},e.prototype.removeObserver=function(e){var t=this.observers_,o=t.indexOf(e);~o&&t.splice(o,1),!t.length&&this.connected_&&this.disconnect_()},e.prototype.refresh=function(){this.updateObservers_()&&this.refresh()},e.prototype.updateObservers_=function(){var e=this.observers_.filter(function(e){return e.gatherActive(),e.hasActive()});return e.forEach(function(e){return e.broadcastActive()}),e.length>0},e.prototype.connect_=function(){tZ&&!this.connected_&&(document.addEventListener("transitionend",this.onTransitionEnd_),window.addEventListener("resize",this.refresh),tN?(this.mutationsObserver_=new MutationObserver(this.refresh),this.mutationsObserver_.observe(document,{attributes:!0,childList:!0,characterData:!0,subtree:!0})):(document.addEventListener("DOMSubtreeModified",this.refresh),this.mutationEventsAdded_=!0),this.connected_=!0)},e.prototype.disconnect_=function(){tZ&&this.connected_&&(document.removeEventListener("transitionend",this.onTransitionEnd_),window.removeEventListener("resize",this.refresh),this.mutationsObserver_&&this.mutationsObserver_.disconnect(),this.mutationEventsAdded_&&document.removeEventListener("DOMSubtreeModified",this.refresh),this.mutationsObserver_=null,this.mutationEventsAdded_=!1,this.connected_=!1)},e.prototype.onTransitionEnd_=function(e){var t=e.propertyName,o=void 0===t?"":t;tD.some(function(e){return!!~o.indexOf(e)})&&this.refresh()},e.getInstance=function(){return this.instance_||(this.instance_=new e),this.instance_},e.instance_=null,e}(),tj=function(e,t){for(var o=0,n=Object.keys(t);o<n.length;o++){var r=n[o];Object.defineProperty(e,r,{value:t[r],enumerable:!1,writable:!1,configurable:!0})}return e},tL=function(e){return e&&e.ownerDocument&&e.ownerDocument.defaultView||tP},tR=tz(0,0,0,0);function t$(e){return parseFloat(e)||0}function tV(e){for(var t=[],o=1;o<arguments.length;o++)t[o-1]=arguments[o];return t.reduce(function(t,o){return t+t$(e["border-"+o+"-width"])},0)}var tH="undefined"!=typeof SVGGraphicsElement?function(e){return e instanceof tL(e).SVGGraphicsElement}:function(e){return e instanceof tL(e).SVGElement&&"function"==typeof e.getBBox};function tz(e,t,o,n){return{x:e,y:t,width:o,height:n}}var tW=function(){function e(e){this.broadcastWidth=0,this.broadcastHeight=0,this.contentRect_=tz(0,0,0,0),this.target=e}return e.prototype.isActive=function(){var e=function(e){if(!tZ)return tR;if(tH(e)){var t;return tz(0,0,(t=e.getBBox()).width,t.height)}return function(e){var t=e.clientWidth,o=e.clientHeight;if(!t&&!o)return tR;var n=tL(e).getComputedStyle(e),r=function(e){for(var t={},o=0,n=["top","right","bottom","left"];o<n.length;o++){var r=n[o],i=e["padding-"+r];t[r]=t$(i)}return t}(n),i=r.left+r.right,a=r.top+r.bottom,u=t$(n.width),c=t$(n.height);if("border-box"===n.boxSizing&&(Math.round(u+i)!==t&&(u-=tV(n,"left","right")+i),Math.round(c+a)!==o&&(c-=tV(n,"top","bottom")+a)),e!==tL(e).document.documentElement){var l=Math.round(u+i)-t,s=Math.round(c+a)-o;1!==Math.abs(l)&&(u-=l),1!==Math.abs(s)&&(c-=s)}return tz(r.left,r.top,u,c)}(e)}(this.target);return this.contentRect_=e,e.width!==this.broadcastWidth||e.height!==this.broadcastHeight},e.prototype.broadcastRect=function(){var e=this.contentRect_;return this.broadcastWidth=e.width,this.broadcastHeight=e.height,e},e}(),tB=function(e,t){var o,n,r,i,a,u=(o=t.x,n=t.y,r=t.width,i=t.height,tj(a=Object.create(("undefined"!=typeof DOMRectReadOnly?DOMRectReadOnly:Object).prototype),{x:o,y:n,width:r,height:i,top:n,right:o+r,bottom:i+n,left:o}),a);tj(this,{target:e,contentRect:u})},tF=function(){function e(e,t,o){if(this.activeObservations_=[],this.observations_=new tT,"function"!=typeof e)throw TypeError("The callback provided as parameter 1 is not a function.");this.callback_=e,this.controller_=t,this.callbackCtx_=o}return e.prototype.observe=function(e){if(!arguments.length)throw TypeError("1 argument required, but only 0 present.");if("undefined"!=typeof Element&&Element instanceof Object){if(!(e instanceof tL(e).Element))throw TypeError('parameter 1 is not of type "Element".');var t=this.observations_;t.has(e)||(t.set(e,new tW(e)),this.controller_.addObserver(this),this.controller_.refresh())}},e.prototype.unobserve=function(e){if(!arguments.length)throw TypeError("1 argument required, but only 0 present.");if("undefined"!=typeof Element&&Element instanceof Object){if(!(e instanceof tL(e).Element))throw TypeError('parameter 1 is not of type "Element".');var t=this.observations_;t.has(e)&&(t.delete(e),t.size||this.controller_.removeObserver(this))}},e.prototype.disconnect=function(){this.clearActive(),this.observations_.clear(),this.controller_.removeObserver(this)},e.prototype.gatherActive=function(){var e=this;this.clearActive(),this.observations_.forEach(function(t){t.isActive()&&e.activeObservations_.push(t)})},e.prototype.broadcastActive=function(){if(this.hasActive()){var e=this.callbackCtx_,t=this.activeObservations_.map(function(e){return new tB(e.target,e.broadcastRect())});this.callback_.call(e,t,e),this.clearActive()}},e.prototype.clearActive=function(){this.activeObservations_.splice(0)},e.prototype.hasActive=function(){return this.activeObservations_.length>0},e}(),tI="undefined"!=typeof WeakMap?new WeakMap:new tT,tY=function e(t){if(!(this instanceof e))throw TypeError("Cannot call a class as a function.");if(!arguments.length)throw TypeError("1 argument required, but only 0 present.");var o=new tF(t,tA.getInstance(),this);tI.set(this,o)};["observe","unobserve","disconnect"].forEach(function(e){tY.prototype[e]=function(){var t;return(t=tI.get(this))[e].apply(t,arguments)}});var tX=void 0!==tP.ResizeObserver?tP.ResizeObserver:tY;function tG(e,t){var o=null,n=null,r=new tX(function(e){var r=(0,H.Z)(e,1)[0].target;if(document.documentElement.contains(r)){var i=r.getBoundingClientRect(),a=i.width,u=i.height,c=Math.floor(a),l=Math.floor(u);(o!==c||n!==l)&&Promise.resolve().then(function(){t({width:c,height:l})}),o=c,n=l}});return e&&r.observe(e),function(){r.disconnect()}}function tq(e){return"function"!=typeof e?null:e()}function tU(e){return"object"===(0,p.Z)(e)&&e?e:null}var tK=m.forwardRef(function(e,t){var o=e.children,n=e.disabled,r=e.target,i=e.align,a=e.onAlign,u=e.monitorWindowResize,c=e.monitorBufferTime,l=m.useRef({}),s=m.useRef(),f=m.Children.only(o),p=m.useRef({});p.current.disabled=n,p.current.target=r,p.current.align=i,p.current.onAlign=a;var d=tM(function(){var e=p.current,t=e.disabled,o=e.target,n=e.align,r=e.onAlign,i=s.current;if(!t&&o&&i){var a,u,c,f,d,h,m,v,g,b,y,w,x=tq(o),E=tU(o);l.current.element=x,l.current.point=E,l.current.align=n;var k=document.activeElement;return x&&t_(x)?w=tC(i,x,n):E&&(f=(c=tl.getDocument(i)).defaultView||c.parentWindow,d=tl.getWindowScrollLeft(f),h=tl.getWindowScrollTop(f),m=tl.viewportWidth(f),v=tl.viewportHeight(f),g={left:a="pageX"in E?E.pageX:d+E.clientX,top:u="pageY"in E?E.pageY:h+E.clientY,width:0,height:0},b=a>=0&&a<=d+m&&u>=0&&u<=h+v,y=[n.points[0],"cc"],w=tk(i,g,eV(eV({},n),{},{points:y}),b)),k!==document.activeElement&&T(i,k)&&"function"==typeof k.focus&&k.focus(),r&&w&&r(i,w),!0}return!1},void 0===c?0:c),h=(0,H.Z)(d,2),v=h[0],g=h[1],b=m.useState(),y=(0,H.Z)(b,2),w=y[0],x=y[1],E=m.useState(),k=(0,H.Z)(E,2),C=k[0],O=k[1];return Y(function(){x(tq(r)),O(tU(r))}),m.useEffect(function(){var e;l.current.element===w&&((e=l.current.point)===C||e&&C&&("pageX"in C&&"pageY"in C?e.pageX===C.pageX&&e.pageY===C.pageY:"clientX"in C&&"clientY"in C&&e.clientX===C.clientX&&e.clientY===C.clientY))&&tO(l.current.align,i)||v()}),m.useEffect(function(){return tG(s.current,v)},[s.current]),m.useEffect(function(){return tG(w,v)},[w]),m.useEffect(function(){n?g():v()},[n]),m.useEffect(function(){if(u)return j(window,"resize",v).remove},[u]),m.useEffect(function(){return function(){g()}},[]),m.useImperativeHandle(t,function(){return{forceAlign:function(){return v(!0)}}}),m.isValidElement(f)&&(f=m.cloneElement(f,{ref:N(f.ref,s)})),f});function tJ(){tJ=function(){return t};var e,t={},o=Object.prototype,n=o.hasOwnProperty,r=Object.defineProperty||function(e,t,o){e[t]=o.value},i="function"==typeof Symbol?Symbol:{},a=i.iterator||"@@iterator",u=i.asyncIterator||"@@asyncIterator",c=i.toStringTag||"@@toStringTag";function l(e,t,o){return Object.defineProperty(e,t,{value:o,enumerable:!0,configurable:!0,writable:!0}),e[t]}try{l({},"")}catch(e){l=function(e,t,o){return e[t]=o}}function s(t,o,n,i){var a,u,c=Object.create((o&&o.prototype instanceof g?o:g).prototype);return r(c,"_invoke",{value:(a=new T(i||[]),u=d,function(o,r){if(u===h)throw Error("Generator is already running");if(u===m){if("throw"===o)throw r;return{value:e,done:!0}}for(a.method=o,a.arg=r;;){var i=a.delegate;if(i){var c=function t(o,n){var r=n.method,i=o.iterator[r];if(i===e)return n.delegate=null,"throw"===r&&o.iterator.return&&(n.method="return",n.arg=e,t(o,n),"throw"===n.method)||"return"!==r&&(n.method="throw",n.arg=TypeError("The iterator does not provide a '"+r+"' method")),v;var a=f(i,o.iterator,n.arg);if("throw"===a.type)return n.method="throw",n.arg=a.arg,n.delegate=null,v;var u=a.arg;return u?u.done?(n[o.resultName]=u.value,n.next=o.nextLoc,"return"!==n.method&&(n.method="next",n.arg=e),n.delegate=null,v):u:(n.method="throw",n.arg=TypeError("iterator result is not an object"),n.delegate=null,v)}(i,a);if(c){if(c===v)continue;return c}}if("next"===a.method)a.sent=a._sent=a.arg;else if("throw"===a.method){if(u===d)throw u=m,a.arg;a.dispatchException(a.arg)}else"return"===a.method&&a.abrupt("return",a.arg);u=h;var l=f(t,n,a);if("normal"===l.type){if(u=a.done?m:"suspendedYield",l.arg===v)continue;return{value:l.arg,done:a.done}}"throw"===l.type&&(u=m,a.method="throw",a.arg=l.arg)}})}),c}function f(e,t,o){try{return{type:"normal",arg:e.call(t,o)}}catch(e){return{type:"throw",arg:e}}}t.wrap=s;var d="suspendedStart",h="executing",m="completed",v={};function g(){}function b(){}function y(){}var w={};l(w,a,function(){return this});var x=Object.getPrototypeOf,E=x&&x(x(Z([])));E&&E!==o&&n.call(E,a)&&(w=E);var k=y.prototype=g.prototype=Object.create(w);function C(e){["next","throw","return"].forEach(function(t){l(e,t,function(e){return this._invoke(t,e)})})}function O(e,t){var o;r(this,"_invoke",{value:function(r,i){function a(){return new t(function(o,a){!function o(r,i,a,u){var c=f(e[r],e,i);if("throw"!==c.type){var l=c.arg,s=l.value;return s&&"object"==(0,p.Z)(s)&&n.call(s,"__await")?t.resolve(s.__await).then(function(e){o("next",e,a,u)},function(e){o("throw",e,a,u)}):t.resolve(s).then(function(e){l.value=e,a(l)},function(e){return o("throw",e,a,u)})}u(c.arg)}(r,i,o,a)})}return o=o?o.then(a,a):a()}})}function _(e){var t={tryLoc:e[0]};1 in e&&(t.catchLoc=e[1]),2 in e&&(t.finallyLoc=e[2],t.afterLoc=e[3]),this.tryEntries.push(t)}function M(e){var t=e.completion||{};t.type="normal",delete t.arg,e.completion=t}function T(e){this.tryEntries=[{tryLoc:"root"}],e.forEach(_,this),this.reset(!0)}function Z(t){if(t||""===t){var o=t[a];if(o)return o.call(t);if("function"==typeof t.next)return t;if(!isNaN(t.length)){var r=-1,i=function o(){for(;++r<t.length;)if(n.call(t,r))return o.value=t[r],o.done=!1,o;return o.value=e,o.done=!0,o};return i.next=i}}throw TypeError((0,p.Z)(t)+" is not iterable")}return b.prototype=y,r(k,"constructor",{value:y,configurable:!0}),r(y,"constructor",{value:b,configurable:!0}),b.displayName=l(y,c,"GeneratorFunction"),t.isGeneratorFunction=function(e){var t="function"==typeof e&&e.constructor;return!!t&&(t===b||"GeneratorFunction"===(t.displayName||t.name))},t.mark=function(e){return Object.setPrototypeOf?Object.setPrototypeOf(e,y):(e.__proto__=y,l(e,c,"GeneratorFunction")),e.prototype=Object.create(k),e},t.awrap=function(e){return{__await:e}},C(O.prototype),l(O.prototype,u,function(){return this}),t.AsyncIterator=O,t.async=function(e,o,n,r,i){void 0===i&&(i=Promise);var a=new O(s(e,o,n,r),i);return t.isGeneratorFunction(o)?a:a.next().then(function(e){return e.done?e.value:a.next()})},C(k),l(k,c,"Generator"),l(k,a,function(){return this}),l(k,"toString",function(){return"[object Generator]"}),t.keys=function(e){var t=Object(e),o=[];for(var n in t)o.push(n);return o.reverse(),function e(){for(;o.length;){var n=o.pop();if(n in t)return e.value=n,e.done=!1,e}return e.done=!0,e}},t.values=Z,T.prototype={constructor:T,reset:function(t){if(this.prev=0,this.next=0,this.sent=this._sent=e,this.done=!1,this.delegate=null,this.method="next",this.arg=e,this.tryEntries.forEach(M),!t)for(var o in this)"t"===o.charAt(0)&&n.call(this,o)&&!isNaN(+o.slice(1))&&(this[o]=e)},stop:function(){this.done=!0;var e=this.tryEntries[0].completion;if("throw"===e.type)throw e.arg;return this.rval},dispatchException:function(t){if(this.done)throw t;var o=this;function r(n,r){return u.type="throw",u.arg=t,o.next=n,r&&(o.method="next",o.arg=e),!!r}for(var i=this.tryEntries.length-1;i>=0;--i){var a=this.tryEntries[i],u=a.completion;if("root"===a.tryLoc)return r("end");if(a.tryLoc<=this.prev){var c=n.call(a,"catchLoc"),l=n.call(a,"finallyLoc");if(c&&l){if(this.prev<a.catchLoc)return r(a.catchLoc,!0);if(this.prev<a.finallyLoc)return r(a.finallyLoc)}else if(c){if(this.prev<a.catchLoc)return r(a.catchLoc,!0)}else{if(!l)throw Error("try statement without catch or finally");if(this.prev<a.finallyLoc)return r(a.finallyLoc)}}}},abrupt:function(e,t){for(var o=this.tryEntries.length-1;o>=0;--o){var r=this.tryEntries[o];if(r.tryLoc<=this.prev&&n.call(r,"finallyLoc")&&this.prev<r.finallyLoc){var i=r;break}}i&&("break"===e||"continue"===e)&&i.tryLoc<=t&&t<=i.finallyLoc&&(i=null);var a=i?i.completion:{};return a.type=e,a.arg=t,i?(this.method="next",this.next=i.finallyLoc,v):this.complete(a)},complete:function(e,t){if("throw"===e.type)throw e.arg;return"break"===e.type||"continue"===e.type?this.next=e.arg:"return"===e.type?(this.rval=this.arg=e.arg,this.method="return",this.next="end"):"normal"===e.type&&t&&(this.next=t),v},finish:function(e){for(var t=this.tryEntries.length-1;t>=0;--t){var o=this.tryEntries[t];if(o.finallyLoc===e)return this.complete(o.completion,o.afterLoc),M(o),v}},catch:function(e){for(var t=this.tryEntries.length-1;t>=0;--t){var o=this.tryEntries[t];if(o.tryLoc===e){var n=o.completion;if("throw"===n.type){var r=n.arg;M(o)}return r}}throw Error("illegal catch attempt")},delegateYield:function(t,o,n){return this.delegate={iterator:Z(t),resultName:o,nextLoc:n},"next"===this.method&&(this.arg=e),v}},t}function tQ(e,t,o,n,r,i,a){try{var u=e[i](a),c=u.value}catch(e){return void o(e)}u.done?t(c):Promise.resolve(c).then(n,r)}tK.displayName="Align";var t0=["measure","alignPre","align",null,"motion"],t1=function(e,t){var o=X(null),n=(0,H.Z)(o,2),r=n[0],i=n[1],a=(0,m.useRef)();function u(){M.cancel(a.current)}return(0,m.useEffect)(function(){i("measure",!0)},[e]),(0,m.useEffect)(function(){if("measure"===r&&t(),r){var e;a.current=M((e=tJ().mark(function e(){var t,o;return tJ().wrap(function(e){for(;;)switch(e.prev=e.next){case 0:t=t0.indexOf(r),(o=t0[t+1])&&-1!==t&&i(o,!0);case 3:case"end":return e.stop()}},e)}),function(){var t=this,o=arguments;return new Promise(function(n,r){var i=e.apply(t,o);function a(e){tQ(i,n,r,a,u,"next",e)}function u(e){tQ(i,n,r,a,u,"throw",e)}a(void 0)})}))}},[r]),(0,m.useEffect)(function(){return function(){u()}},[]),[r,function(e){u(),a.current=M(function(){i(function(e){switch(r){case"align":return"motion";case"motion":return"stable"}return e},!0),null==e||e()})}]},t2=function(e){var t=m.useState({width:0,height:0}),o=(0,H.Z)(t,2),n=o[0],r=o[1];return[m.useMemo(function(){var t={};if(e){var o=n.width,r=n.height;-1!==e.indexOf("height")&&r?t.height=r:-1!==e.indexOf("minHeight")&&r&&(t.minHeight=r),-1!==e.indexOf("width")&&o?t.width=o:-1!==e.indexOf("minWidth")&&o&&(t.minWidth=o)}return t},[e,n]),function(e){var t=e.offsetWidth,o=e.offsetHeight,n=e.getBoundingClientRect(),i=n.width,a=n.height;1>Math.abs(t-i)&&1>Math.abs(o-a)&&(t=i,o=a),r({width:t,height:o})}]},t5=m.forwardRef(function(e,t){var o=e.visible,n=e.prefixCls,r=e.className,i=e.style,a=e.children,u=e.zIndex,c=e.stretch,l=e.destroyPopupOnHide,s=e.forceRender,p=e.align,h=e.point,v=e.getRootDomNode,g=e.getClassNameFromAlign,b=e.onAlign,y=e.onMouseEnter,w=e.onMouseLeave,x=e.onMouseDown,E=e.onTouchStart,k=e.onClick,C=(0,m.useRef)(),O=(0,m.useRef)(),_=(0,m.useState)(),M=(0,H.Z)(_,2),T=M[0],Z=M[1],P=t2(c),S=(0,H.Z)(P,2),D=S[0],N=S[1],A=t1(o,function(){c&&N(v())}),j=(0,H.Z)(A,2),L=j[0],R=j[1],$=(0,m.useState)(0),z=(0,H.Z)($,2),W=z[0],B=z[1],F=(0,m.useRef)();function I(){var e;null===(e=C.current)||void 0===e||e.forceAlign()}function X(e,t){var o=g(t);T!==o&&Z(o),B(function(e){return e+1}),"align"===L&&(null==b||b(e,t))}Y(function(){"alignPre"===L&&B(0)},[L]),Y(function(){"align"===L&&(W<3?I():R(function(){var e;null===(e=F.current)||void 0===e||e.call(F)}))},[W]);var G=(0,d.Z)({},eL(e));function q(){return new Promise(function(e){F.current=e})}["onAppearEnd","onEnterEnd","onLeaveEnd"].forEach(function(e){var t=G[e];G[e]=function(e,o){return R(),null==t?void 0:t(e,o)}}),m.useEffect(function(){G.motionName||"motion"!==L||R()},[G.motionName,L]),m.useImperativeHandle(t,function(){return{forceAlign:I,getElement:function(){return O.current}}});var U=(0,d.Z)((0,d.Z)({},D),{},{zIndex:u,opacity:"motion"!==L&&"stable"!==L&&o?0:void 0,pointerEvents:o||"stable"===L?void 0:"none"},i),K=!0;null!=p&&p.points&&("align"===L||"stable"===L)&&(K=!1);var J=a;return m.Children.count(a)>1&&(J=m.createElement("div",{className:"".concat(n,"-content")},a)),m.createElement(eM,(0,f.Z)({visible:o,ref:O,leavedClassName:"".concat(n,"-hidden")},G,{onAppearPrepare:q,onEnterPrepare:q,removeOnLeave:l,forceRender:s}),function(e,t){var o=e.className,i=e.style,a=V()(n,r,T,o);return m.createElement(tK,{target:h||v,key:"popup",ref:C,monitorWindowResize:!0,disabled:K,align:p,onAlign:X},m.createElement("div",{ref:t,className:a,onMouseEnter:y,onMouseLeave:w,onMouseDownCapture:x,onTouchStartCapture:E,onClick:k,style:(0,d.Z)((0,d.Z)({},i),U)},J))})});t5.displayName="PopupInner";var t4=m.forwardRef(function(e,t){var o=e.prefixCls,n=e.visible,r=e.zIndex,i=e.children,a=e.mobile,u=(a=void 0===a?{}:a).popupClassName,c=a.popupStyle,l=a.popupMotion,s=a.popupRender,p=e.onClick,h=m.useRef();m.useImperativeHandle(t,function(){return{forceAlign:function(){},getElement:function(){return h.current}}});var v=(0,d.Z)({zIndex:r},c),g=i;return m.Children.count(i)>1&&(g=m.createElement("div",{className:"".concat(o,"-content")},i)),s&&(g=s(g)),m.createElement(eM,(0,f.Z)({visible:n,ref:h,removeOnLeave:!0},void 0===l?{}:l),function(e,t){var n=e.className,r=e.style,i=V()(o,u,n);return m.createElement("div",{ref:t,className:i,onClick:p,style:(0,d.Z)((0,d.Z)({},r),v)},g)})});t4.displayName="MobilePopupInner";var t3=["visible","mobile"],t9=m.forwardRef(function(e,t){var o=e.visible,n=e.mobile,r=(0,h.Z)(e,t3),i=(0,m.useState)(o),a=(0,H.Z)(i,2),u=a[0],c=a[1],l=(0,m.useState)(!1),s=(0,H.Z)(l,2),p=s[0],v=s[1],g=(0,d.Z)((0,d.Z)({},r),{},{visible:u});(0,m.useEffect)(function(){c(o),o&&n&&v(z())},[o,n]);var b=p?m.createElement(t4,(0,f.Z)({},g,{mobile:n,ref:t})):m.createElement(t5,(0,f.Z)({},g,{ref:t}));return m.createElement("div",null,m.createElement(eR,g),b)});t9.displayName="Popup";var t7=m.createContext(null);function t6(){}var t8=["onClick","onMouseDown","onTouchStart","onMouseEnter","onMouseLeave","onFocus","onBlur","onContextMenu"],oe=(c=function(e){(0,y.Z)(o,e);var t=(0,w.Z)(o);function o(e){var n,r;return(0,v.Z)(this,o),n=t.call(this,e),(0,x.Z)((0,b.Z)(n),"popupRef",m.createRef()),(0,x.Z)((0,b.Z)(n),"triggerRef",m.createRef()),(0,x.Z)((0,b.Z)(n),"portalContainer",void 0),(0,x.Z)((0,b.Z)(n),"attachId",void 0),(0,x.Z)((0,b.Z)(n),"clickOutsideHandler",void 0),(0,x.Z)((0,b.Z)(n),"touchOutsideHandler",void 0),(0,x.Z)((0,b.Z)(n),"contextMenuOutsideHandler1",void 0),(0,x.Z)((0,b.Z)(n),"contextMenuOutsideHandler2",void 0),(0,x.Z)((0,b.Z)(n),"mouseDownTimeout",void 0),(0,x.Z)((0,b.Z)(n),"focusTime",void 0),(0,x.Z)((0,b.Z)(n),"preClickTime",void 0),(0,x.Z)((0,b.Z)(n),"preTouchTime",void 0),(0,x.Z)((0,b.Z)(n),"delayTimer",void 0),(0,x.Z)((0,b.Z)(n),"hasPopupMouseDown",void 0),(0,x.Z)((0,b.Z)(n),"onMouseEnter",function(e){var t=n.props.mouseEnterDelay;n.fireEvents("onMouseEnter",e),n.delaySetPopupVisible(!0,t,t?null:e)}),(0,x.Z)((0,b.Z)(n),"onMouseMove",function(e){n.fireEvents("onMouseMove",e),n.setPoint(e)}),(0,x.Z)((0,b.Z)(n),"onMouseLeave",function(e){n.fireEvents("onMouseLeave",e),n.delaySetPopupVisible(!1,n.props.mouseLeaveDelay)}),(0,x.Z)((0,b.Z)(n),"onPopupMouseEnter",function(){n.clearDelayTimer()}),(0,x.Z)((0,b.Z)(n),"onPopupMouseLeave",function(e){var t;e.relatedTarget&&!e.relatedTarget.setTimeout&&T(null===(t=n.popupRef.current)||void 0===t?void 0:t.getElement(),e.relatedTarget)||n.delaySetPopupVisible(!1,n.props.mouseLeaveDelay)}),(0,x.Z)((0,b.Z)(n),"onFocus",function(e){n.fireEvents("onFocus",e),n.clearDelayTimer(),n.isFocusToShow()&&(n.focusTime=Date.now(),n.delaySetPopupVisible(!0,n.props.focusDelay))}),(0,x.Z)((0,b.Z)(n),"onMouseDown",function(e){n.fireEvents("onMouseDown",e),n.preClickTime=Date.now()}),(0,x.Z)((0,b.Z)(n),"onTouchStart",function(e){n.fireEvents("onTouchStart",e),n.preTouchTime=Date.now()}),(0,x.Z)((0,b.Z)(n),"onBlur",function(e){n.fireEvents("onBlur",e),n.clearDelayTimer(),n.isBlurToHide()&&n.delaySetPopupVisible(!1,n.props.blurDelay)}),(0,x.Z)((0,b.Z)(n),"onContextMenu",function(e){e.preventDefault(),n.fireEvents("onContextMenu",e),n.setPopupVisible(!0,e)}),(0,x.Z)((0,b.Z)(n),"onContextMenuClose",function(){n.isContextMenuToShow()&&n.close()}),(0,x.Z)((0,b.Z)(n),"onClick",function(e){if(n.fireEvents("onClick",e),n.focusTime){var t;if(n.preClickTime&&n.preTouchTime?t=Math.min(n.preClickTime,n.preTouchTime):n.preClickTime?t=n.preClickTime:n.preTouchTime&&(t=n.preTouchTime),20>Math.abs(t-n.focusTime))return;n.focusTime=0}n.preClickTime=0,n.preTouchTime=0,n.isClickToShow()&&(n.isClickToHide()||n.isBlurToHide())&&e&&e.preventDefault&&e.preventDefault();var o=!n.state.popupVisible;(n.isClickToHide()&&!o||o&&n.isClickToShow())&&n.setPopupVisible(!n.state.popupVisible,e)}),(0,x.Z)((0,b.Z)(n),"onPopupMouseDown",function(){if(n.hasPopupMouseDown=!0,clearTimeout(n.mouseDownTimeout),n.mouseDownTimeout=window.setTimeout(function(){n.hasPopupMouseDown=!1},0),n.context){var e;(e=n.context).onPopupMouseDown.apply(e,arguments)}}),(0,x.Z)((0,b.Z)(n),"onDocumentClick",function(e){if(!n.props.mask||n.props.maskClosable){var t=e.target,o=n.getRootDomNode(),r=n.getPopupDomNode();(!T(o,t)||n.isContextMenuOnly())&&!T(r,t)&&!n.hasPopupMouseDown&&n.close()}}),(0,x.Z)((0,b.Z)(n),"getRootDomNode",function(){var e=n.props.getTriggerDOMNode;if(e)return e(n.triggerRef.current);try{var t=P(n.triggerRef.current);if(t)return t}catch(e){}return E.findDOMNode((0,b.Z)(n))}),(0,x.Z)((0,b.Z)(n),"getPopupClassNameFromAlign",function(e){var t=[],o=n.props,r=o.popupPlacement,i=o.builtinPlacements,a=o.prefixCls,u=o.alignPoint,c=o.getPopupClassNameFromAlign;return r&&i&&t.push(function(e,t,o,n){for(var r=o.points,i=Object.keys(e),a=0;a<i.length;a+=1){var u,c=i[a];if(u=e[c].points,n?u[0]===r[0]:u[0]===r[0]&&u[1]===r[1])return"".concat(t,"-placement-").concat(c)}return""}(i,a,e,u)),c&&t.push(c(e)),t.join(" ")}),(0,x.Z)((0,b.Z)(n),"getComponent",function(){var e=n.props,t=e.prefixCls,o=e.destroyPopupOnHide,r=e.popupClassName,i=e.onPopupAlign,a=e.popupMotion,u=e.popupAnimation,c=e.popupTransitionName,l=e.popupStyle,s=e.mask,p=e.maskAnimation,d=e.maskTransitionName,h=e.maskMotion,v=e.zIndex,g=e.popup,b=e.stretch,y=e.alignPoint,w=e.mobile,x=e.forceRender,E=e.onPopupClick,k=n.state,C=k.popupVisible,O=k.point,_=n.getPopupAlign(),M={};return n.isMouseEnterToShow()&&(M.onMouseEnter=n.onPopupMouseEnter),n.isMouseLeaveToHide()&&(M.onMouseLeave=n.onPopupMouseLeave),M.onMouseDown=n.onPopupMouseDown,M.onTouchStart=n.onPopupMouseDown,m.createElement(t9,(0,f.Z)({prefixCls:t,destroyPopupOnHide:o,visible:C,point:y&&O,className:r,align:_,onAlign:i,animation:u,getClassNameFromAlign:n.getPopupClassNameFromAlign},M,{stretch:b,getRootDomNode:n.getRootDomNode,style:l,mask:s,zIndex:v,transitionName:c,maskAnimation:p,maskTransitionName:d,maskMotion:h,ref:n.popupRef,motion:a,mobile:w,forceRender:x,onClick:E}),"function"==typeof g?g():g)}),(0,x.Z)((0,b.Z)(n),"attachParent",function(e){M.cancel(n.attachId);var t,o=n.props,r=o.getPopupContainer,i=o.getDocument,a=n.getRootDomNode();r?(a||0===r.length)&&(t=r(a)):t=i(n.getRootDomNode()).body,t?t.appendChild(e):n.attachId=M(function(){n.attachParent(e)})}),(0,x.Z)((0,b.Z)(n),"getContainer",function(){if(!n.portalContainer){var e=(0,n.props.getDocument)(n.getRootDomNode()).createElement("div");e.style.position="absolute",e.style.top="0",e.style.left="0",e.style.width="100%",n.portalContainer=e}return n.attachParent(n.portalContainer),n.portalContainer}),(0,x.Z)((0,b.Z)(n),"setPoint",function(e){n.props.alignPoint&&e&&n.setState({point:{pageX:e.pageX,pageY:e.pageY}})}),(0,x.Z)((0,b.Z)(n),"handlePortalUpdate",function(){n.state.prevPopupVisible!==n.state.popupVisible&&n.props.afterPopupVisibleChange(n.state.popupVisible)}),(0,x.Z)((0,b.Z)(n),"triggerContextValue",{onPopupMouseDown:n.onPopupMouseDown}),r="popupVisible"in e?!!e.popupVisible:!!e.defaultPopupVisible,n.state={prevPopupVisible:r,popupVisible:r},t8.forEach(function(e){n["fire".concat(e)]=function(t){n.fireEvents(e,t)}}),n}return(0,g.Z)(o,[{key:"componentDidMount",value:function(){this.componentDidUpdate()}},{key:"componentDidUpdate",value:function(){var e,t=this.props;if(this.state.popupVisible){!this.clickOutsideHandler&&(this.isClickToHide()||this.isContextMenuToShow())&&(e=t.getDocument(this.getRootDomNode()),this.clickOutsideHandler=j(e,"mousedown",this.onDocumentClick)),this.touchOutsideHandler||(e=e||t.getDocument(this.getRootDomNode()),this.touchOutsideHandler=j(e,"touchstart",this.onDocumentClick)),!this.contextMenuOutsideHandler1&&this.isContextMenuToShow()&&(e=e||t.getDocument(this.getRootDomNode()),this.contextMenuOutsideHandler1=j(e,"scroll",this.onContextMenuClose)),!this.contextMenuOutsideHandler2&&this.isContextMenuToShow()&&(this.contextMenuOutsideHandler2=j(window,"blur",this.onContextMenuClose));return}this.clearOutsideHandler()}},{key:"componentWillUnmount",value:function(){this.clearDelayTimer(),this.clearOutsideHandler(),clearTimeout(this.mouseDownTimeout),M.cancel(this.attachId)}},{key:"getPopupDomNode",value:function(){var e;return(null===(e=this.popupRef.current)||void 0===e?void 0:e.getElement())||null}},{key:"getPopupAlign",value:function(){var e,t=this.props,o=t.popupPlacement,n=t.popupAlign,r=t.builtinPlacements;return o&&r?(e=r[o]||{},(0,d.Z)((0,d.Z)({},e),n)):n}},{key:"setPopupVisible",value:function(e,t){var o=this.props.alignPoint,n=this.state.popupVisible;this.clearDelayTimer(),n!==e&&("popupVisible"in this.props||this.setState({popupVisible:e,prevPopupVisible:n}),this.props.onPopupVisibleChange(e)),o&&t&&e&&this.setPoint(t)}},{key:"delaySetPopupVisible",value:function(e,t,o){var n=this,r=1e3*t;if(this.clearDelayTimer(),r){var i=o?{pageX:o.pageX,pageY:o.pageY}:null;this.delayTimer=window.setTimeout(function(){n.setPopupVisible(e,i),n.clearDelayTimer()},r)}else this.setPopupVisible(e,o)}},{key:"clearDelayTimer",value:function(){this.delayTimer&&(clearTimeout(this.delayTimer),this.delayTimer=null)}},{key:"clearOutsideHandler",value:function(){this.clickOutsideHandler&&(this.clickOutsideHandler.remove(),this.clickOutsideHandler=null),this.contextMenuOutsideHandler1&&(this.contextMenuOutsideHandler1.remove(),this.contextMenuOutsideHandler1=null),this.contextMenuOutsideHandler2&&(this.contextMenuOutsideHandler2.remove(),this.contextMenuOutsideHandler2=null),this.touchOutsideHandler&&(this.touchOutsideHandler.remove(),this.touchOutsideHandler=null)}},{key:"createTwoChains",value:function(e){var t=this.props.children.props,o=this.props;return t[e]&&o[e]?this["fire".concat(e)]:t[e]||o[e]}},{key:"isClickToShow",value:function(){var e=this.props,t=e.action,o=e.showAction;return -1!==t.indexOf("click")||-1!==o.indexOf("click")}},{key:"isContextMenuOnly",value:function(){var e=this.props.action;return"contextMenu"===e||1===e.length&&"contextMenu"===e[0]}},{key:"isContextMenuToShow",value:function(){var e=this.props,t=e.action,o=e.showAction;return -1!==t.indexOf("contextMenu")||-1!==o.indexOf("contextMenu")}},{key:"isClickToHide",value:function(){var e=this.props,t=e.action,o=e.hideAction;return -1!==t.indexOf("click")||-1!==o.indexOf("click")}},{key:"isMouseEnterToShow",value:function(){var e=this.props,t=e.action,o=e.showAction;return -1!==t.indexOf("hover")||-1!==o.indexOf("mouseEnter")}},{key:"isMouseLeaveToHide",value:function(){var e=this.props,t=e.action,o=e.hideAction;return -1!==t.indexOf("hover")||-1!==o.indexOf("mouseLeave")}},{key:"isFocusToShow",value:function(){var e=this.props,t=e.action,o=e.showAction;return -1!==t.indexOf("focus")||-1!==o.indexOf("focus")}},{key:"isBlurToHide",value:function(){var e=this.props,t=e.action,o=e.hideAction;return -1!==t.indexOf("focus")||-1!==o.indexOf("blur")}},{key:"forcePopupAlign",value:function(){if(this.state.popupVisible){var e;null===(e=this.popupRef.current)||void 0===e||e.forceAlign()}}},{key:"fireEvents",value:function(e,t){var o=this.props.children.props[e];o&&o(t);var n=this.props[e];n&&n(t)}},{key:"close",value:function(){this.setPopupVisible(!1)}},{key:"render",value:function(){var e,t=this.state.popupVisible,o=this.props,n=o.children,r=o.forceRender,i=o.alignPoint,a=o.className,u=o.autoDestroy,c=m.Children.only(n),l={key:"trigger"};this.isContextMenuToShow()?l.onContextMenu=this.onContextMenu:l.onContextMenu=this.createTwoChains("onContextMenu"),this.isClickToHide()||this.isClickToShow()?(l.onClick=this.onClick,l.onMouseDown=this.onMouseDown,l.onTouchStart=this.onTouchStart):(l.onClick=this.createTwoChains("onClick"),l.onMouseDown=this.createTwoChains("onMouseDown"),l.onTouchStart=this.createTwoChains("onTouchStart")),this.isMouseEnterToShow()?(l.onMouseEnter=this.onMouseEnter,i&&(l.onMouseMove=this.onMouseMove)):l.onMouseEnter=this.createTwoChains("onMouseEnter"),this.isMouseLeaveToHide()?l.onMouseLeave=this.onMouseLeave:l.onMouseLeave=this.createTwoChains("onMouseLeave"),this.isFocusToShow()||this.isBlurToHide()?(l.onFocus=this.onFocus,l.onBlur=this.onBlur):(l.onFocus=this.createTwoChains("onFocus"),l.onBlur=this.createTwoChains("onBlur"));var s=V()(c&&c.props&&c.props.className,a);s&&(l.className=s);var f=(0,d.Z)({},l);A(c)&&(f.ref=N(this.triggerRef,c.ref));var p=m.cloneElement(c,f);return(t||this.popupRef.current||r)&&(e=m.createElement(R,{key:"portal",getContainer:this.getContainer,didUpdate:this.handlePortalUpdate},this.getComponent())),!t&&u&&(e=null),m.createElement(t7.Provider,{value:this.triggerContextValue},p,e)}}],[{key:"getDerivedStateFromProps",value:function(e,t){var o=e.popupVisible,n={};return void 0!==o&&t.popupVisible!==o&&(n.popupVisible=o,n.prevPopupVisible=t.popupVisible),n}}]),o}(m.Component),(0,x.Z)(c,"contextType",t7),(0,x.Z)(c,"defaultProps",{prefixCls:"rc-trigger-popup",getPopupClassNameFromAlign:function(){return""},getDocument:function(e){return e?e.ownerDocument:window.document},onPopupVisibleChange:t6,afterPopupVisibleChange:t6,onPopupAlign:t6,popupClassName:"",mouseEnterDelay:0,mouseLeaveDelay:.1,focusDelay:0,blurDelay:.15,popupStyle:{},destroyPopupOnHide:!1,popupAlign:{},defaultPopupVisible:!1,mask:!1,maskClosable:!0,action:[],showAction:[],hideAction:[],autoDestroy:!1}),c),ot={adjustX:1,adjustY:1},oo=[0,0],on={left:{points:["cr","cl"],overflow:ot,offset:[-4,0],targetOffset:oo},right:{points:["cl","cr"],overflow:ot,offset:[4,0],targetOffset:oo},top:{points:["bc","tc"],overflow:ot,offset:[0,-4],targetOffset:oo},bottom:{points:["tc","bc"],overflow:ot,offset:[0,4],targetOffset:oo},topLeft:{points:["bl","tl"],overflow:ot,offset:[0,-4],targetOffset:oo},leftTop:{points:["tr","tl"],overflow:ot,offset:[-4,0],targetOffset:oo},topRight:{points:["br","tr"],overflow:ot,offset:[0,-4],targetOffset:oo},rightTop:{points:["tl","tr"],overflow:ot,offset:[4,0],targetOffset:oo},bottomRight:{points:["tr","br"],overflow:ot,offset:[0,4],targetOffset:oo},rightBottom:{points:["bl","br"],overflow:ot,offset:[4,0],targetOffset:oo},bottomLeft:{points:["tl","bl"],overflow:ot,offset:[0,4],targetOffset:oo},leftBottom:{points:["br","bl"],overflow:ot,offset:[-4,0],targetOffset:oo}};function or(e){var t=e.showArrow,o=e.arrowContent,n=e.children,r=e.prefixCls,i=e.id,a=e.overlayInnerStyle,u=e.className,c=e.style;return m.createElement("div",{className:V()("".concat(r,"-content"),u),style:c},!1!==t&&m.createElement("div",{className:"".concat(r,"-arrow"),key:"arrow"},o),m.createElement("div",{className:"".concat(r,"-inner"),id:i,role:"tooltip",style:a},"function"==typeof n?n():n))}var oi=["overlayClassName","trigger","mouseEnterDelay","mouseLeaveDelay","overlayStyle","prefixCls","children","onVisibleChange","afterVisibleChange","transitionName","animation","motion","placement","align","destroyTooltipOnHide","defaultVisible","getTooltipContainer","overlayInnerStyle","arrowContent","overlay","id","showArrow"],oa=(0,m.forwardRef)(function(e,t){var o=e.overlayClassName,n=e.trigger,r=e.mouseEnterDelay,i=e.mouseLeaveDelay,a=e.overlayStyle,u=e.prefixCls,c=void 0===u?"rc-tooltip":u,l=e.children,s=e.onVisibleChange,v=e.afterVisibleChange,g=e.transitionName,b=e.animation,y=e.motion,w=e.placement,x=e.align,E=e.destroyTooltipOnHide,k=void 0!==E&&E,C=e.defaultVisible,O=e.getTooltipContainer,_=e.overlayInnerStyle,M=e.arrowContent,T=e.overlay,Z=e.id,P=e.showArrow,S=void 0===P||P,D=(0,h.Z)(e,oi),N=(0,m.useRef)(null);(0,m.useImperativeHandle)(t,function(){return N.current});var A=(0,d.Z)({},D);"visible"in e&&(A.popupVisible=e.visible);var j=!1,L=!1;if("boolean"==typeof k)j=k;else if(k&&"object"===(0,p.Z)(k)){var R=k.keepParent;j=!0===R,L=!1===R}return m.createElement(oe,(0,f.Z)({popupClassName:o,prefixCls:c,popup:function(){return m.createElement(or,{showArrow:S,arrowContent:M,key:"content",prefixCls:c,id:Z,overlayInnerStyle:_},T)},action:void 0===n?["hover"]:n,builtinPlacements:on,popupPlacement:void 0===w?"right":w,ref:N,popupAlign:void 0===x?{}:x,getPopupContainer:O,onPopupVisibleChange:s,afterPopupVisibleChange:v,popupTransitionName:g,popupAnimation:b,popupMotion:y,defaultPopupVisible:C,destroyPopupOnHide:j,autoDestroy:L,mouseLeaveDelay:void 0===i?.1:i,popupStyle:a,mouseEnterDelay:void 0===r?0:r},A),l)})},51162:function(e,t){var o=Symbol.for("react.element"),n=Symbol.for("react.portal"),r=Symbol.for("react.fragment"),i=Symbol.for("react.strict_mode"),a=Symbol.for("react.profiler"),u=Symbol.for("react.provider"),c=Symbol.for("react.context"),l=Symbol.for("react.server_context"),s=Symbol.for("react.forward_ref"),f=Symbol.for("react.suspense"),p=Symbol.for("react.suspense_list"),d=Symbol.for("react.memo"),h=Symbol.for("react.lazy");function m(e){if("object"==typeof e&&null!==e){var t=e.$$typeof;switch(t){case o:switch(e=e.type){case r:case a:case i:case f:case p:return e;default:switch(e=e&&e.$$typeof){case l:case c:case s:case h:case d:case u:return e;default:return t}}case n:return t}}}Symbol.for("react.offscreen"),Symbol.for("react.module.reference"),t.ForwardRef=s,t.isFragment=function(e){return m(e)===r},t.isMemo=function(e){return m(e)===d}},11805:function(e,t,o){e.exports=o(51162)},50029:function(e,t,o){function n(e,t,o,n,r,i,a){try{var u=e[i](a),c=u.value}catch(e){o(e);return}u.done?t(c):Promise.resolve(c).then(n,r)}function r(e){return function(){var t=this,o=arguments;return new Promise(function(r,i){var a=e.apply(t,o);function u(e){n(a,r,i,u,c,"next",e)}function c(e){n(a,r,i,u,c,"throw",e)}u(void 0)})}}o.d(t,{Z:function(){return r}})}}]);