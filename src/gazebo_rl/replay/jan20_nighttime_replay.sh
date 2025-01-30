# accept the start and end idxs from the command line
# and replay the episodes from start to end

# list of integers 
trials=(113 121 122 124 125 126 128 129 130 131 132 137 138 139 143 146 148 149 151 154 156 157 158 160 161 162 164 165 166 167 168 171 172 173 175 176 179 181 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 255 256 257 258 259 260 261 262 263 265 266 267 269 273 274 275 276 277 278 279 280 281 283 284 286 290 297 298 299 301 302 303 304 305 306 308 309 311 316 317 318 319 320 321 325 326 327 328 329 330 331 332 333 334 335)


for i in "${trials[@]}"
do
    # keep the leading 0
    echo "Replaying episode $i"
    # python publish_saved_video.py -d user_$i -na -v -c 700 -clo 200
    python construct_fastrl_dataset.py -d user_$i -na -c 700 -clo 200 -r 0.70

    # if the core dumps, then exit
    if [ $? -ne 0 ]; then
        echo "Core dumped. Exiting."
        exit
    fi
done
