# accept the start and end idxs from the command line
# and replay the episodes from start to end

# list of integers 
trials=(113 121 122 124 125 126 128 129 130 131 132 137 138 139 140 143 145 146 148 149 151 153 154 156 157 158 160 161 162 164 165 166 167 168 171 172 173 175 176 179 181 335 333 332 330 329 328 327 326 325 321 320 319 318 317 316 311 309 308 306 305 304 303 302 301 299 298 297 290 286 284 283 281 280 279 278 277 276 275 274 273 269 267 266 265 263 262 261 259 258 257 256 255 252 251 250 249 248 247 246 245 244 243 242 239 237 236 235 234 233 232)


for i in "${trials[@]}"
do
    # keep the leading 0
    echo "Replaying episode $i"
    # python publish_saved_video.py -d user_$i -na -v -c 700 -clo 200
    python construct_fastrl_dataset.py -d user_$i -na -v -c 700 -clo 200
done
