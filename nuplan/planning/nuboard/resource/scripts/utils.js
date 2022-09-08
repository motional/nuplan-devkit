function openTab(evt, tab_name) {
  var i, tab_content, header_bar_tab_btn;
  tab_content = document.getElementsByClassName("tab-content");
  const navEles = document.getElementsByClassName("nav-item");

  for (const navEle of navEles) {
    const navName = navEle.innerText;
    navEle.style.fontWeight =
      tab_name === navName.toLowerCase() ? "bold" : "unset";
  }
  for (i = 0; i < tab_content.length; i++) {
    tab_content[i].style.display = "none";
  }
  document.getElementById(tab_name).style.display = "block";
  evt.currentTarget.className += " active";
}

function toggleNav() {
  const navbar = document.getElementsByClassName("navbar")[0];
  const menuIcon = document.getElementById("menu-icon");
  const file_header = document.getElementById("file-header");

  // Close the menu bar
  if (file_header.style.marginLeft == "10rem" || file_header.style.marginLeft == "") {
      navbar.style.marginLeft = "-10rem";
      file_header.style.marginLeft = 0;
      menuIcon.classList.add("icon-menu");
      menuIcon.classList.remove("icon-arrow-left");
  } else {
      navbar.style.marginLeft = 0;
      file_header.style.marginLeft = "10rem";
      menuIcon.classList.add("icon-arrow-left");
      menuIcon.classList.remove("icon-menu");
  }
}

function openScenarioBar(evt, bar_name, svg_path_name) {
  const scenario_bar = document.getElementById(bar_name);
  const scenario_bar_svg_path = document.getElementById(svg_path_name);
  if (scenario_bar.style.display == "none" || scenario_bar.style.display == "") {
        scenario_bar.style.display = "block";
        scenario_bar_svg_path.setAttribute("d", "M12 8l-6 6 1.41 1.41L12 10.83l4.59 4.58L18 14z");
   } else {
        scenario_bar.style.display = "none";
        scenario_bar_svg_path.setAttribute("d", "M16.59 8.59L12 13.17 7.41 8.59 6 10l6 6 6-6z");
   }
}

function openModal(evt, modal_name) {
  const overlay = document.getElementsByClassName("overlay")[0];
  overlay.style.display = "block";

  document.getElementsByClassName(modal_name)[0].style.display = "block";
}

function closeModal(evt, modal_name) {
  const overlay = document.getElementsByClassName("overlay")[0];
  overlay.style.display = "none";

  document.getElementsByClassName(modal_name)[0].style.display = "none";
}
