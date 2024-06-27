import{S as H,i as V,s as z,E as L,G as X,k as B,a as F,l as R,m as O,h as y,c as j,_ as J,n as D,b as P,H as w,I,X as M,J as Q,K as Y,L as Z,g as G,d as K,W as x,ah as q,N as A,O as S,w as W,q as T,r as U,an as $,y as ee,z as te,A as ae,ao as le,B as ne}from"../chunks/index.6e58b9bb.js";import{e as se}from"../chunks/dialogs.cd7ae541.js";import{d as oe,l as ie}from"../chunks/technologicStores.3ea6c695.js";function ue(a){let e;return{c(){e=T("Select a File")},l(t){e=U(t,"Select a File")},m(t,o){P(t,e,o)},d(t){t&&y(e)}}}function ce(a){let e,t,o,p,s,c,g,_,u,h,f,d=[{type:"file"},{name:a[1]},a[6]()],b={};for(let n=0;n<d.length;n+=1)b=L(b,d[n]);const v=a[11].default,E=X(v,a,a[10],null),m=E||ue();return{c(){e=B("div"),t=B("div"),o=B("input"),p=F(),s=B("button"),m&&m.c(),this.h()},l(n){e=R(n,"DIV",{class:!0,"data-testid":!0});var l=O(e);t=R(l,"DIV",{class:!0});var r=O(t);o=R(r,"INPUT",{type:!0,name:!0}),r.forEach(y),p=j(l),s=R(l,"BUTTON",{type:!0,class:!0});var k=O(s);m&&m.l(k),k.forEach(y),l.forEach(y),this.h()},h(){J(o,b),D(t,"class","w-0 h-0 overflow-hidden"),D(s,"type","button"),D(s,"class",c="file-button-btn "+a[3]),s.disabled=g=a[7].disabled,D(e,"class",_="file-button "+a[4]),D(e,"data-testid","file-button")},m(n,l){P(n,e,l),w(e,t),w(t,o),o.autofocus&&o.focus(),a[16](o),w(e,p),w(e,s),m&&m.m(s,null),u=!0,h||(f=[I(o,"change",a[17]),I(o,"change",a[15]),I(s,"click",a[5]),I(s,"keydown",a[12]),I(s,"keyup",a[13]),I(s,"keypress",a[14])],h=!0)},p(n,[l]){J(o,b=M(d,[{type:"file"},(!u||l&2)&&{name:n[1]},n[6]()])),E&&E.p&&(!u||l&1024)&&Q(E,v,n,n[10],u?Z(v,n[10],l,null):Y(n[10]),null),(!u||l&8&&c!==(c="file-button-btn "+n[3]))&&D(s,"class",c),(!u||l&128&&g!==(g=n[7].disabled))&&(s.disabled=g),(!u||l&16&&_!==(_="file-button "+n[4]))&&D(e,"class",_)},i(n){u||(G(m,n),u=!0)},o(n){K(m,n),u=!1},d(n){n&&y(e),a[16](null),m&&m.d(n),h=!1,x(f)}}}const fe="btn";function re(a,e,t){let o,p;const s=["files","name","width","button"];let c=q(e,s),{$$slots:g={},$$scope:_}=e,{files:u=void 0}=e,{name:h}=e,{width:f=""}=e,{button:d="variant-filled"}=e,b;function v(){b.click()}function E(){return delete c.class,c}function m(i){S.call(this,a,i)}function n(i){S.call(this,a,i)}function l(i){S.call(this,a,i)}function r(i){S.call(this,a,i)}function k(i){W[i?"unshift":"push"](()=>{b=i,t(2,b)})}function N(){u=this.files,t(0,u)}return a.$$set=i=>{t(18,e=L(L({},e),A(i))),t(7,c=q(e,s)),"files"in i&&t(0,u=i.files),"name"in i&&t(1,h=i.name),"width"in i&&t(8,f=i.width),"button"in i&&t(9,d=i.button),"$$scope"in i&&t(10,_=i.$$scope)},a.$$.update=()=>{t(4,o=`${e.class??""}`),a.$$.dirty&768&&t(3,p=`${fe} ${d} ${f}`)},e=A(e),[u,h,b,p,o,v,E,c,f,d,_,g,m,n,l,r,k,N]}class de extends H{constructor(e){super(),V(this,e,re,ce,z,{files:0,name:1,width:8,button:9})}}function be(a){let e;return{c(){e=T("Reload Database")},l(t){e=U(t,"Reload Database")},m(t,o){P(t,e,o)},d(t){t&&y(e)}}}function _e(a){let e,t,o,p,s,c,g,_,u,h,f,d,b,v,E;function m(l){a[5](l)}let n={name:"dbfiles",button:"variant-filled-secondary",disabled:!a[2],$$slots:{default:[be]},$$scope:{ctx:a}};return a[1]!==void 0&&(n.files=a[1]),f=new de({props:n}),W.push(()=>$(f,"files",m)),f.$on("change",a[4]),{c(){e=B("section"),t=B("h3"),o=T("Backup / Restore"),p=F(),s=B("p"),c=B("button"),g=T("Download Database"),u=F(),h=B("p"),ee(f.$$.fragment),this.h()},l(l){e=R(l,"SECTION",{class:!0});var r=O(e);t=R(r,"H3",{});var k=O(t);o=U(k,"Backup / Restore"),k.forEach(y),p=j(r),s=R(r,"P",{});var N=O(s);c=R(N,"BUTTON",{class:!0});var i=O(c);g=U(i,"Download Database"),i.forEach(y),N.forEach(y),u=j(r),h=R(r,"P",{});var C=O(h);te(f.$$.fragment,C),C.forEach(y),r.forEach(y),this.h()},h(){D(c,"class","btn variant-filled-primary"),c.disabled=_=!a[0],D(e,"class","card p-3 m-3 variant-glass flex flex-col gap-2")},m(l,r){P(l,e,r),w(e,t),w(t,o),w(e,p),w(e,s),w(s,c),w(c,g),w(e,u),w(e,h),ae(f,h,null),b=!0,v||(E=I(c,"click",a[3]),v=!0)},p(l,[r]){(!b||r&1&&_!==(_=!l[0]))&&(c.disabled=_);const k={};r&4&&(k.disabled=!l[2]),r&64&&(k.$$scope={dirty:r,ctx:l}),!d&&r&2&&(d=!0,k.files=l[1],le(()=>d=!1)),f.$set(k)},i(l){b||(G(f.$$.fragment,l),b=!0)},o(l){K(f.$$.fragment,l),b=!1},d(l){l&&y(e),ne(f),v=!1,E()}}}function he(a,e,t){let o=!0;async function p(){t(0,o=!1);const u=JSON.stringify(await oe(),null,2),h=new Blob([u],{type:"application/octet-stream"}),f=window.URL.createObjectURL(h),d=document.createElement("a");d.href=f,d.download="technologic.json",d.click(),window.URL.revokeObjectURL(f),t(0,o=!0)}let s,c=!0;async function g(u){return t(2,c=!1),await se("Are you sure you want to load the database file? It will overwrite the current database. This action cannot be undone.")&&(await ie(JSON.parse(await s[0].text())),window.location.reload()),t(2,c=!0),!1}function _(u){s=u,t(1,s)}return[o,s,c,p,g,_]}class ke extends H{constructor(e){super(),V(this,e,he,_e,z,{})}}export{ke as component};