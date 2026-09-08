---
layout: default
title: Writing
---

# Writing

Notes on things I've worked on, mostly one post per paper or project.

<ul class="news">
{% for post in site.posts %}
  <li><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y-%m-%d" }}</time>: <a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>
